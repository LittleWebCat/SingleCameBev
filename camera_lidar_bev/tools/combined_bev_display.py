#!/usr/bin/python3
"""Shared camera/BEV renderer and a CPU-only viewer of saved predictions."""
import argparse
import json
import math
from pathlib import Path
import textwrap

import cv2
import numpy as np


BACKGROUND = (24, 19, 14)
TEXT = (238, 232, 222)
MUTED = (181, 164, 144)
CAMERA_2D = (70, 240, 245)
LIDAR_3D = (40, 165, 255)
MATCHED = (90, 245, 100)
ROAD = (75, 155, 65)
LANE = (240, 70, 230)
ROAD_EDGE = (255, 110, 45)
LOCAL_MAP = (150, 110, 55)
LOCAL_PATH = (220, 185, 75)
DEFAULT_LAYERS = dict(points=True, projection=True, camera2d=True,
                      lidar3d=True, labels=True,
                      road=True, lanes=True, boundaries=True, mapping=True)


def calibration_arrays(calibration):
    return (np.asarray(calibration['K'], dtype=float).reshape(3, 3),
            np.asarray(calibration['D'], dtype=float),
            np.asarray(calibration['T'], dtype=float).reshape(4, 4))


def load_calibration(intrinsics_file, extrinsics_file):
    import yaml
    camera = yaml.safe_load(Path(intrinsics_file).read_text())
    extrinsics = yaml.safe_load(Path(extrinsics_file).read_text())
    return dict(width=camera['image_width'], height=camera['image_height'],
                K=camera['camera_matrix']['data'],
                D=camera['distortion_coefficients']['data'],
                T=np.asarray(extrinsics['transforms']['T_camera_from_os_sensor']['matrix']).reshape(-1).tolist())


def cuboid(center, dimensions, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    signs = np.array([[(1 if i & bit else -1) for bit in (1, 2, 4)] for i in range(8)])
    return (signs*np.asarray(dimensions)/2)@rotation.T+np.asarray(center)


def radial_limit(distortion):
    """Same first positive radial fold boundary used by the native projection."""
    d = np.pad(np.asarray(distortion), (0, max(0, 5-len(distortion))))
    limits = []
    for coefficients in ([d[4], d[1], d[0], 1.],
                         [7*d[4], 5*d[1], 3*d[0], 1.]):
        for root in np.roots(np.trim_zeros(coefficients, 'f')):
            if root.real > 0 and abs(root.imag) <= 1e-7*max(1., abs(root.real)):
                limits.append(root.real)
    return min(limits, default=float('inf'))


def project_camera(points, calibration):
    k, d, _ = calibration_arrays(calibration)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    mask = np.isfinite(points).all(axis=1) & (points[:, 2] > .3)
    radius = np.full(len(points), np.inf)
    radius[mask] = np.sum((points[mask, :2]/points[mask, 2:])**2, axis=1)
    mask &= radius < radial_limit(d)
    pixels = np.full((len(points), 2), np.nan)
    if mask.any():
        pixels[mask] = cv2.projectPoints(points[mask], np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
        mask &= np.isfinite(pixels).all(axis=1)
    return pixels, mask


def projected_box(box, calibration):
    _, _, transform = calibration_arrays(calibration)
    center = np.asarray(box['os_sensor_xyz_m'])
    corners = cuboid(center, box['size_lwh_m'], box['yaw_rad'])
    points = np.vstack((center, corners))@transform[:3, :3].T+transform[:3, 3]
    pixels, valid = project_camera(points, calibration)
    width, height = calibration['width'], calibration['height']
    visible = bool(valid.all() and 0 <= pixels[0, 0] < width and 0 <= pixels[0, 1] < height)
    if not visible:
        return dict(visible=False, native_xyxy_px=[0.]*4, native_center_px=[0.]*2)
    lower = np.maximum(pixels[1:].min(axis=0), [0, 0])
    upper = np.minimum(pixels[1:].max(axis=0), [width, height])
    return dict(visible=bool((upper > lower).all()),
                native_xyxy_px=[*lower.tolist(), *upper.tolist()],
                native_center_px=pixels[0].tolist())


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def associations_for_sample(sample, calibration):
    """Apply the current native matching criteria to display-only box data."""
    lidar, camera = sample.get('lidar_boxes', []), sample.get('camera_2d_boxes', [])
    rows = [dict(lidar_index=i, camera_index=-1, iou=0.,
                 projection=projected_box(box, calibration)) for i, box in enumerate(lidar)]
    scores = np.full((len(lidar), len(camera)), -1.)
    for i, row in enumerate(rows):
        projection = row['projection']
        if not projection['visible']:
            continue
        u, v = projection['native_center_px']
        for j, box in enumerate(camera):
            roi = box['native_xyxy_px']
            compatible = (box['class'] == 'person' if lidar[i]['class'] == 'person'
                          else lidar[i]['class'] == 'car' and box['class'] in ('car', 'motorcycle', 'bus', 'truck'))
            eligible = (.35 <= box['score'] <= 1 and np.isfinite(roi).all() and
                        0 <= roi[0] < roi[2] <= calibration['width'] and
                        0 <= roi[1] < roi[3] <= calibration['height'])
            if not compatible or not eligible or not (roi[0] <= u <= roi[2] and roi[1] <= v <= roi[3]):
                continue
            iou = overlap(projection['native_xyxy_px'], roi)
            if iou >= .2:
                scores[i, j] = iou
    def unique(values):
        if not len(values):
            return -1
        order = np.argsort(values)[::-1]
        return int(order[0]) if values[order[0]] >= 0 and (len(order) == 1 or values[order[1]] < 0 or values[order[0]]-values[order[1]] > .05) else -1
    for i, row in enumerate(rows):
        j = unique(scores[i])
        if j >= 0 and unique(scores[:, j]) == i:
            row.update(camera_index=j, iou=float(scores[i, j]))
    return rows


def text(canvas, label, position, color=TEXT, scale=.5, background=False):
    x, y = map(int, position)
    (width, height), baseline = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(2, min(x, canvas.shape[1]-width-3))
    y = max(height+3, min(y, canvas.shape[0]-baseline-3))
    if background:
        cv2.rectangle(canvas, (x-2, y-height-2), (x+width+2, y+baseline+2), BACKGROUND, -1)
    cv2.putText(canvas, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def wire(canvas, pixels, valid, color, thickness=2):
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j <= i or not (valid[i] and valid[j]):
                continue
            a, b = tuple(np.clip(pixels[i], -10000, 10000).astype(int)), tuple(np.clip(pixels[j], -10000, 10000).astype(int))
            visible, a, b = cv2.clipLine((0, 0, canvas.shape[1], canvas.shape[0]), a, b)
            if visible:
                cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


class CombinedBevRenderer:
    def __init__(self, calibration, forward_m=80., lateral_m=30., max_points=30000):
        self.calibration = calibration
        self.k, self.d, self.transform = calibration_arrays(calibration)
        self.forward_m, self.lateral_m, self.max_points = forward_m, lateral_m, max_points

    def render(self, image_bgr, sample, points=None, status=None, layers=None, segmentation=None,
               mapping=None):
        layers = {**DEFAULT_LAYERS, **(layers or {})}
        display_status = dict(status or sample.get('live_status', {}))
        image = image_bgr.copy()
        bev_size = image.shape[0]
        bev = np.full((bev_size, bev_size, 3), BACKGROUND, dtype=np.uint8)
        inset, side = 35, bev_size-65
        # Fit the metric rectangle in the square panel with one shared scale.
        # A metre along X and Y must occupy the same number of pixels so that
        # extended forward ranges preserve object dimensions and headings.
        pixels_per_m = min(side/self.forward_m, side/(2*self.lateral_m))
        metric_width = 2*self.lateral_m*pixels_per_m
        metric_height = self.forward_m*pixels_per_m
        grid_left = inset+(side-metric_width)/2
        grid_top = inset+(side-metric_height)/2
        grid_right = grid_left+metric_width
        grid_bottom = grid_top+metric_height
        def bev_pixels(xy):
            xy = np.asarray(xy)
            return np.stack((grid_left+pixels_per_m*(self.lateral_m-xy[..., 1]),
                             grid_bottom-pixels_per_m*xy[..., 0]), axis=-1)
        for forward in np.arange(0, self.forward_m+.01, 10):
            y = int(round(bev_pixels([forward, 0])[1]))
            cv2.line(bev, (round(grid_left), y), (round(grid_right), y), (61, 50, 40), 1)
            text(bev, f'{forward:g}', (grid_left-30, y+4), MUTED, .4)
        for lateral in np.arange(-self.lateral_m, self.lateral_m+.01, 10):
            x = int(round(bev_pixels([0, lateral])[0]))
            cv2.line(bev, (x, round(grid_top)), (x, round(grid_bottom)), (61, 50, 40), 1)
            text(bev, f'{lateral:g}', (x-8, grid_bottom+24), MUTED, .4)
        text(bev, f'Forward X 0-{self.forward_m:g} m | left +Y +/-{self.lateral_m:g} m | os_sensor', (inset, 21), TEXT, .48)
        if layers['mapping'] and mapping is not None:
            # Mapping is already transformed using scan-paired local odometry.
            # It is a BEV backdrop only: never project accumulated map returns
            # onto the camera or transform the source-frame detector boxes.
            local_map = np.asarray(mapping.get('points', np.empty((0, 3))), dtype=float).reshape(-1, 3)
            keep = (np.isfinite(local_map).all(axis=1) & (local_map[:, 0] >= 0) &
                    (local_map[:, 0] <= self.forward_m) &
                    (np.abs(local_map[:, 1]) <= self.lateral_m))
            if keep.any():
                pixels = np.rint(bev_pixels(local_map[keep, :2])).astype(int)
                valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < bev_size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < bev_size)
                bev[pixels[valid, 1], pixels[valid, 0]] = LOCAL_MAP
        source_points = np.empty((0, 3))
        point_count = 0
        projected_count = 0
        if points is not None and len(points):
            source_points = np.asarray(points)[:, :3]
            point_count = len(source_points)
            keep = (np.isfinite(source_points).all(axis=1) & (source_points[:, 0] >= 0) &
                    (source_points[:, 0] <= self.forward_m) & (np.abs(source_points[:, 1]) <= self.lateral_m) &
                    (np.linalg.norm(source_points, axis=1) >= .5))
            source_points = source_points[keep]
            source_points = source_points[::max(1, math.ceil(len(source_points)/self.max_points))]
            if layers['points'] and len(source_points):
                pixels = np.rint(bev_pixels(source_points[:, :2])).astype(int)
                valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < bev_size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < bev_size)
                bev[pixels[valid, 1], pixels[valid, 0]] = (135, 132, 124)
            if layers['projection'] and len(source_points):
                camera_points = source_points@self.transform[:3, :3].T+self.transform[:3, 3]
                pixels, valid = project_camera(camera_points, self.calibration)
                valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
                selected = np.flatnonzero(valid)
                # Keep the nearest sampled return at each pixel.
                order = selected[np.argsort(camera_points[selected, 2])]
                quantized = np.rint(pixels[order]).astype(int)
                quantized[:, 0] = np.clip(quantized[:, 0], 0, image.shape[1]-1)
                quantized[:, 1] = np.clip(quantized[:, 1], 0, image.shape[0]-1)
                _, first = np.unique(quantized[:, 1]*image.shape[1]+quantized[:, 0], return_index=True)
                quantized = quantized[first]
                projected_count = len(quantized)
                image[quantized[:, 1], quantized[:, 0]] = (
                    image[quantized[:, 1], quantized[:, 0]]*.35+np.array([220, 220, 220])*.65).astype(np.uint8)
        road_info = 'Camera-image road/lane masks and edges: waiting or unavailable | BEV road geometry disabled'
        if segmentation is not None:
            road_mask = np.asarray(segmentation['road_mask']) > 0
            lane_mask = np.asarray(segmentation['lane_mask']) > 0
            if road_mask.shape != image.shape[:2] or lane_mask.shape != image.shape[:2]:
                raise ValueError('Road/lane masks must match the displayed native image')
            if layers['road']:
                image[road_mask] = (image[road_mask]*.78+np.asarray(ROAD)*.22).astype(np.uint8)
            if layers['lanes']:
                image[lane_mask] = (image[lane_mask]*.18+np.asarray(LANE)*.82).astype(np.uint8)
            if layers['boundaries']:
                contours, _ = cv2.findContours(road_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        continue
                    line = contour.reshape(-1, 2)
                    # Native image borders are clipping limits, not road edges.
                    inside = ((line[:, 0] > 2) & (line[:, 0] < image.shape[1]-3) &
                              (line[:, 1] > 2) & (line[:, 1] < image.shape[0]-3))
                    for i in range(len(line)):
                        j = (i+1) % len(line)
                        if inside[i] and inside[j]:
                            cv2.line(image, tuple(line[i]), tuple(line[j]), ROAD_EDGE, 2, cv2.LINE_AA)
            road_info = 'Camera-image road/lane masks and semantic edges only | BEV road geometry disabled'
        associations = sample.get('associations')
        if associations is None:
            associations = associations_for_sample(sample, self.calibration)
        matched = {a['lidar_index']: a for a in associations if a['camera_index'] >= 0 and
                   sample.get('lidar_boxes', [])[a['lidar_index']].get('camera_matched', True)}
        camera_matches = {a['camera_index']: a for a in matched.values()}
        records = []

        def footprint(corners, center, ident, cls, score, dims, color, yaw):
            if not (0 <= center[0] <= self.forward_m and abs(center[1]) <= self.lateral_m):
                return
            hull = cv2.convexHull(np.asarray(corners[:, :2], np.float32)).reshape(-1, 2)
            pixels = np.rint(bev_pixels(hull)).astype(int)
            cv2.polylines(bev, [pixels], True, color, 2, cv2.LINE_AA)
            heading = np.array([center[:2], center[:2]+np.array([math.cos(yaw), math.sin(yaw)])*max(1., dims[0]/2)])
            direction = np.rint(bev_pixels(heading)).astype(int)
            cv2.arrowedLine(bev, tuple(direction[0]), tuple(direction[1]), color, 2, cv2.LINE_AA, tipLength=.25)
            if layers['labels']:
                p = np.rint(bev_pixels(center[:2])).astype(int)
                text(bev, f'{ident} {cls} {score:.2f}', (p[0]+5, p[1]-8), color, .42, True)
                text(bev, f'{center[0]:.1f}m | {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}', (p[0]+5, p[1]+10), color, .38, True)

        for i, box in enumerate(sample.get('lidar_boxes', [])):
            center, dims, yaw = np.asarray(box['os_sensor_xyz_m']), box['size_lwh_m'], box['yaw_rad']
            corners = cuboid(center, dims, yaw)
            production_match = box.get('camera_matched', i in matched)
            color = MATCHED if production_match else LIDAR_3D
            ident = f'L{i}'
            note = (f'C{matched[i]["camera_index"]} IoU {matched[i]["iou"]:.2f}' if i in matched else
                    'production camera match' if production_match else 'no 2D match')
            if layers['lidar3d']:
                footprint(corners, center, ident, box['class'], box['score'], dims, color, yaw)
                pixels, valid = project_camera(corners@self.transform[:3, :3].T+self.transform[:3, 3], self.calibration)
                wire(image, pixels, valid, color)
                if valid.any() and layers['labels']:
                    position = np.nanmin(pixels[valid], axis=0)
                    text(image, f'{ident} LiDAR {box["class"]} {box["score"]:.2f}', position-[0, 8], color, .42, True)
            records.append((ident, box['class'], box['score'], center, dims, math.degrees(yaw), note, color))
        for i, box in enumerate(sample.get('camera_2d_boxes', [])):
            roi = np.asarray(box['native_xyxy_px'])
            ident = f'C{i}'
            if layers['camera2d']:
                cv2.rectangle(image, tuple(np.rint(roi[:2]).astype(int)), tuple(np.rint(roi[2:]).astype(int)), CAMERA_2D, 2)
                if layers['labels']:
                    text(image, f'{ident} YOLO {box["class"]} {box["score"]:.2f}', roi[:2]-[0, 8], CAMERA_2D, .42, True)
                if i in camera_matches:
                    center = sample['lidar_boxes'][camera_matches[i]['lidar_index']]['os_sensor_xyz_m']
                    p = np.rint(bev_pixels(np.asarray(center[:2]))).astype(int)
                    if 0 <= p[0] < bev_size and 0 <= p[1] < bev_size:
                        cv2.circle(bev, tuple(p), 6, CAMERA_2D, 2)
                        if layers['labels']:
                            text(bev, ident+' YOLO match', p+[5, 25], CAMERA_2D, .4, True)
            if i in camera_matches:
                lidar_index = camera_matches[i]['lidar_index']
                center = sample['lidar_boxes'][lidar_index]['os_sensor_xyz_m']
                note = f'BEV via L{lidar_index}; pixels '+','.join(f'{n:.0f}' for n in roi)
            else:
                center = None
                note = 'image only; pixels '+','.join(f'{n:.0f}' for n in roi)
            records.append((ident, box['class'], box['score'], center, None, None, note, CAMERA_2D))

        if layers['mapping'] and mapping is not None:
            # Draw trajectory last so raw returns and detector footprints do
            # not erase it. All geometry is restricted to the BEV panel.
            path = np.asarray(mapping.get('path', np.empty((0, 3))), dtype=float).reshape(-1, 3)
            clip = (round(grid_left), round(grid_top), max(1, round(metric_width)), max(1, round(metric_height)))
            if len(path) > 1:
                starts, ends = path[:-1, :2], path[1:, :2]
                low, high = np.minimum(starts, ends), np.maximum(starts, ends)
                intersects = (np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1) &
                              (high[:, 0] >= 0) & (low[:, 0] <= self.forward_m) &
                              (high[:, 1] >= -self.lateral_m) & (low[:, 1] <= self.lateral_m))
                segments = np.stack((starts[intersects], ends[intersects]), axis=1)
                path_pixels = np.clip(np.rint(bev_pixels(segments)), -100000, 100000).astype(int)
                for pixels in path_pixels:
                    visible, start, end = cv2.clipLine(clip, tuple(pixels[0]), tuple(pixels[1]))
                    if visible:
                        cv2.line(bev, start, end, LOCAL_PATH, 2, cv2.LINE_AA)

            # An independently scaled inset includes past motion behind the
            # rig; it leaves the main 80 m metric projection unchanged.
            world_path = np.asarray(mapping.get('world_path', path), dtype=float).reshape(-1, 3)
            world_path = world_path[np.isfinite(world_path).all(axis=1)]
            position = np.asarray(mapping.get('world_position', [0., 0., 0.]), dtype=float).reshape(3)
            world_path = np.vstack((world_path, position))
            inset_size = min(240 if mapping.get('live_localization') else 180,
                             max(90, bev_size//3))
            left, upper = round(grid_right)-inset_size-7, round(grid_bottom)-inset_size-7
            cv2.rectangle(bev, (left, upper), (left+inset_size, upper+inset_size), (27, 25, 21), -1)
            cv2.rectangle(bev, (left, upper), (left+inset_size, upper+inset_size), LOCAL_PATH, 1)
            inset_title = ('LIVE POSITION / MAP' if mapping.get('live_localization') else
                           'GPS ENU TRAJECTORY' if mapping.get('georeferenced') else 'LOCAL TRAJECTORY')
            text(bev, inset_title,
                 (left+7, upper+16), LOCAL_PATH, .36)
            overview = np.asarray(mapping.get('overview_points', np.empty((0, 3))), float).reshape(-1, 3)
            overview = overview[np.isfinite(overview).all(axis=1)]
            bounds = np.vstack((world_path[:, :2], overview[:, :2]))
            lower, higher = bounds.min(axis=0), bounds.max(axis=0)
            center = (lower+higher)/2
            span_m = max(5., float((higher-lower).max()))
            scale = (inset_size-36)/span_m
            middle = np.array([left+inset_size/2, upper+inset_size/2])
            def world_pixels(xy):
                return np.rint((np.asarray(xy)-center)*[scale, -scale]+middle).astype(int)
            if len(overview):
                overview_pixels = world_pixels(overview[:, :2])
                valid = ((overview_pixels[:, 0] > left+3) & (overview_pixels[:, 0] < left+inset_size-3) &
                         (overview_pixels[:, 1] > upper+23) & (overview_pixels[:, 1] < upper+inset_size-20))
                bev[overview_pixels[valid, 1], overview_pixels[valid, 0]] = LOCAL_MAP
            path_pixels = world_pixels(world_path[:, :2])
            if len(path_pixels) > 1:
                cv2.polylines(bev, [path_pixels], False, LOCAL_PATH, 2, cv2.LINE_AA)
            cv2.circle(bev, tuple(path_pixels[0]), 3, MUTED, -1)
            current = path_pixels[-1]
            yaw = float(mapping.get('world_yaw_rad', math.pi/2))
            heading = current+np.rint([math.cos(yaw)*12, -math.sin(yaw)*12]).astype(int)
            cv2.arrowedLine(bev, tuple(current), tuple(heading), MATCHED, 2, cv2.LINE_AA, tipLength=.35)
            cv2.circle(bev, tuple(current), 3, MATCHED, -1)
            if layers['lidar3d']:
                for i, box in enumerate(mapping.get('world_boxes', [])):
                    world_corners = np.asarray(box['corners_world_m'], float).reshape(8, 3)
                    hull = cv2.convexHull(world_corners[:, :2].astype(np.float32)).reshape(-1, 2)
                    color = MATCHED if box.get('camera_matched') else LIDAR_3D
                    pixels = world_pixels(hull)
                    clip = (left+3, upper+23, inset_size-6, inset_size-43)
                    for a, b in zip(pixels, np.roll(pixels, -1, axis=0)):
                        visible, a, b = cv2.clipLine(clip, tuple(a), tuple(b))
                        if visible:
                            cv2.line(bev, a, b, color, 2, cv2.LINE_AA)
                    object_position = world_pixels(np.asarray(box['world_xyz_m'])[:2])
                    if (left+3 <= object_position[0] <= left+inset_size-3 and
                            upper+23 <= object_position[1] <= upper+inset_size-20):
                        cv2.circle(bev, tuple(object_position), 2, color, -1)
                        if layers['labels'] and i < 6:
                            text(bev, f'L{i} {box["class"]}', object_position+[4, -5], color, .3, True)
            coordinates = 'ENU east/north' if mapping.get('georeferenced') else 'local map XY'
            text(bev, f'{coordinates} | span {span_m:.0f} m', (left+7, upper+inset_size-7), MUTED, .31)

        map_state = 'ON' if layers['mapping'] else 'OFF'
        readiness = display_status.get('mapping readiness',
                                       'ready' if mapping is not None else 'no mapping data | start --mapping')
        badge_color = LOCAL_PATH if layers['mapping'] else MUTED
        badge_x, badge_y = round(grid_left)+7, round(grid_top)+18
        source_label = (' LIVE' if mapping is not None and mapping.get('live_localization') else
                        ' SAVED' if display_status.get('map source') else '')
        ready_label = ('GPS ALIGNED' if mapping is not None and mapping.get('georeferenced') else
                       'READY' if mapping is not None else 'WAITING')
        text(bev, f'MAP{source_label} {map_state} [9]  |  '+ready_label,
             (badge_x, badge_y), badge_color, .48, True)
        for i, line in enumerate(textwrap.wrap(str(readiness), width=max(24, int(metric_width/7)))):
            if i >= 2:
                break
            text(bev, line, (badge_x, badge_y+18*(i+1)), badge_color, .36, True)

        status_items = [f'{key}: {value}' for key, value in display_status.items()]
        status_lines = textwrap.wrap(' | '.join(status_items) or 'Saved predictions | CPU display only', width=230)
        top, margin, gap = 168, 22, 26
        width = image.shape[1]+bev_size+2*margin+gap
        table_top = top+image.shape[0]+55
        height = table_top+max(1, len(records))*24+len(status_lines)*23+63
        canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
        text(canvas, 'CAMERA IMAGE + LiDAR PROJECTION   |   FLAT BEV WITH LiDAR BOXES + CAMERA LABELS', (margin, 30), TEXT, .7)
        text(canvas, 'Yellow: YOLO 2D   Orange: LiDAR 3D   Green: LiDAR + YOLO match   Gray: LiDAR returns', (margin, 55), TEXT, .49)
        native_count = sample.get('input_points', point_count)
        sample_info = f'Frame {sample.get("sample_id", "?")} | LiDAR input {native_count:,} / display {point_count:,} / drawn {len(source_points):,} / image {projected_count:,} | accepted matches {len(matched)}'
        sample_info += f' | boxes 2D {len(sample.get("camera_2d_boxes", []))} / LiDAR {len(sample.get("lidar_boxes", []))}'
        if sample.get('pair_delta_ms') is not None:
            sample_info += f' | camera-LiDAR header difference {sample["pair_delta_ms"]:.1f} ms'
        text(canvas, sample_info, (margin, 78), MUTED, .45)
        text(canvas, f'Source header ns | camera {sample.get("image_stamp_ns", 0)} | LiDAR {sample.get("lidar_stamp_ns", 0)}', (margin, 101), MUTED, .43)
        text(canvas, 'Camera image only: green predicted road mask | magenta predicted lane mask | blue semantic road-mask edges', (margin, 124), TEXT, .44)
        text(canvas, road_info, (margin, 146), MUTED, .44)
        canvas[top:top+image.shape[0], margin:margin+image.shape[1]] = image
        canvas[top:top+bev_size, margin+image.shape[1]+gap:margin+image.shape[1]+gap+bev_size] = bev
        columns = (margin, 100, 230, 315, 620, 880, 1040)
        for name, x in zip(('ID/source', 'Class', 'Confidence', 'Center X,Y,Z in os_sensor (m)', 'L x W x H (m)', 'Yaw (deg)', 'Association / image pixels'), columns):
            text(canvas, name, (x, table_top-10), MUTED, .42)
        for row, (ident, cls, score, center, dims, yaw, note, color) in enumerate(records):
            values = (ident, cls, f'{score:.3f}', ', '.join(f'{n:+.2f}' for n in center) if center is not None else '--',
                      ' x '.join(f'{n:.2f}' for n in dims) if dims is not None else '--',
                      f'{yaw:+.1f}' if yaw is not None else '--', note)
            y = table_top+row*24+13
            if row % 2 == 0:
                cv2.rectangle(canvas, (margin-5, y-17), (width-margin, y+6), (35, 29, 23), -1)
            for value, x in zip(values, columns):
                text(canvas, value, (x, y), color, .43)
        tail = table_top+max(1, len(records))*24
        for i, line in enumerate(status_lines):
            text(canvas, line, (margin, tail+20+i*23), MUTED, .42)
        tail += len(status_lines)*23
        text(canvas, 'Frame-local IDs; LiDAR box dimensions are predictions. Camera 2D boxes gain BEV position only through an accepted LiDAR match.', (margin, tail+20), MUTED, .40)
        text(canvas, f'Layers: 1 points  2 projection  3 camera 2D  4 LiDAR boxes  5 labels  6 road mask  7 lane mask  8 image edges  9 MAP {map_state} | arrows: samples | q: close', (margin, tail+41), MUTED, .40)
        return canvas


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root/'evaluation/perception_speed_compare_001.json')
    parser.add_argument('--manifest', type=Path, default=root/'data/perception_compare_001/manifest.json')
    parser.add_argument('--sample-id', default='002')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--forward-m', type=float, default=80.)
    parser.add_argument('--lateral-m', type=float, default=30.)
    parser.add_argument('--max-points', type=int, default=30000)
    args = parser.parse_args()
    if not args.output and not args.display:
        parser.error('Use --output to save a preview or --display to browse cached samples')
    if args.output and args.output.exists():
        parser.error('Choose a new output filename')
    if min(args.forward_m, args.lateral_m, args.max_points) <= 0:
        parser.error('Ranges and max-points must be positive')
    results, manifest = json.loads(args.results.read_text()), json.loads(args.manifest.read_text())
    rows = results['samples']
    sources = {s['sample_id']: s for s in manifest['samples']}
    index = next((i for i, s in enumerate(rows) if s['sample_id'] == args.sample_id), None)
    if index is None:
        parser.error('Sample ID is absent from saved results')
    renderer = CombinedBevRenderer(manifest['calibration'], args.forward_m, args.lateral_m, args.max_points)
    layers = DEFAULT_LAYERS.copy()
    def render(index):
        sample = rows[index]
        source = sources[sample['sample_id']]
        image = cv2.imread(source['image'])
        if image is None:
            raise RuntimeError('Cannot read cached camera image')
        points = np.fromfile(source['cloud'], np.float32).reshape(-1, 4)
        runs = [r for r in results.get('timing_runs', []) if r['sample_id'] == sample['sample_id']]
        status = {'source': 'cached predictions', 'alignment': 'saved candidate B'}
        for key, name in [('lidar_processing_ms', 'PointPillars'), ('yolo_processing_ms', 'YOLO')]:
            if runs:
                status[name] = f'{np.median([r[key] for r in runs]):.1f} ms'
        return renderer.render(image, sample, points, status, layers)
    canvas = render(index)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), canvas):
            raise RuntimeError('Cannot save combined view')
        print(json.dumps({'output': str(args.output.resolve()), 'sample_id': rows[index]['sample_id'],
                          'inference_run': False, 'accuracy_validation_run': False}))
    if args.display:
        name = 'Camera + LiDAR + BEV | cached samples'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1500, 1000)
        while True:
            cv2.imshow(name, canvas)
            key = cv2.waitKeyEx(50)
            if key in (ord('q'), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('n'), 83, 65363, 2555904):
                index = (index+1) % len(rows)
            elif key in (ord('p'), 81, 65361, 2424832):
                index = (index-1) % len(rows)
            elif key in range(ord('1'), ord('1')+len(DEFAULT_LAYERS)):
                layer = list(DEFAULT_LAYERS)[key-ord('1')]
                layers[layer] = not layers[layer]
            else:
                continue
            canvas = render(index)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
