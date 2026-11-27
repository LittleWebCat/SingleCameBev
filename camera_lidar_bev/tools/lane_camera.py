#!/usr/bin/env python3
"""Camera-only lane lines, road boundaries and lane centre, live from the front cameras (no LiDAR, no map).

For each front camera (front_narrow, front_wide) runs TwinLiteNet
(models/road_lanes_001/twinlitenet_fp16.engine, BDD100K lane + drivable-area masks) on its newest image and places
mask pixels on the road with that camera's calibration (config/calibration/rig_20261125) and a flat road plane at
--road-z in os_sensor (the LiDAR frame; about -1.75 m on this car):
  * lane lines: lane-mask pixels projected to the road;
  * road boundary: per image row, the outermost pixels of the drivable-area region in front of the camera,
    ignoring the image border (a region cut by the image edge is not a road edge);
  * lines, ego lane and lane centre: tools/road_features.py, identical for the LiDAR method.
Run one process per camera (--camera front_wide) for low latency; tools/fuse_camera_lanes.py builds camera
combinations from the road points each camera publishes on .../camera_<name>/points. front_wide runs the model on a
centred crop above the bonnet (CROP; drawn as a grey box in the annotated image). The image is resized on
the GPU and the model's masks stay at 640x360 (lane pixels are scaled to the image arithmetically), so the CPU only
receives the image and fits lines. Publishes /camera_lidar_bev/lanes/camera_<name> (JSON), .../camera_<name>_markers
(RViz, os_sensor), .../camera_<name>/segmentation (the road / lane masks on the road plane as a green / blue
point cloud in os_sensor, for the 3D view) and, at --annotate-hz, .../camera_<name>/image: the camera picture with the segmentation (drivable
road green, lane pixels blue), its road edge and fitted lane lines (blue / violet) and the LiDAR's lane lines and road edge
(/camera_lidar_bev/lanes/lidar, orange / red) projected into it.
Run with the workspace .venv Python (TensorRT + torch):

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  ../.venv/bin/python tools/lane_camera.py --camera front_wide [--road-z -1.75] [--annotate-hz 10]
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

import road_features as rf
from lane_lidar import CAMERA_STYLE, markers
from road_lane_detector import TwinLiteRuntime, image_bgr

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / 'config/calibration/rig_20261125'
CAMERAS = {name: (f'/camera/{name}/image_raw', RIG / f'{name}_{size}.yaml', RIG / f'{name}_T_os_sensor.yaml')
           for name, size in (('front_narrow', '1024x768'), ('front_wide', '1920x1080'))}
# Region of the image given to the model, as image-height fractions (top, bottom); width is the centred 16:9 band.
# front_wide: drop the bonnet (below ~70 %) and narrow the 117-degree view towards the dashcam views TwinLiteNet was
# trained on; on 2026-11-25 this doubled the road found and recovered the lane lines the full frame missed.
CROP = {'front_wide': (0.06, 0.70)}


class GroundProjector:
    """Image pixels -> points on the plane z = road_z in os_sensor."""

    def __init__(self, intrinsics, extrinsics, road_z):
        intr = yaml.safe_load(Path(intrinsics).read_text())
        self.k = np.array(intr['camera_matrix']['data'], float).reshape(3, 3)
        self.d = np.array(intr['distortion_coefficients']['data'], float)
        self.size = (intr['image_width'], intr['image_height'])
        cam_from_sensor = np.array(yaml.safe_load(Path(extrinsics).read_text())['transforms']['T_camera_from_os_sensor']['matrix'], float)
        sensor_from_cam = np.linalg.inv(cam_from_sensor)
        self.rot, self.origin, self.road_z = sensor_from_cam[:3, :3], sensor_from_cam[:3, 3], road_z

    def project(self, uv, max_x=45.0):
        if len(uv) == 0:
            return np.zeros((0, 2))
        n = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), self.k, self.d).reshape(-1, 2)
        rays = np.c_[n, np.ones(len(n))] @ self.rot.T
        down = rays[:, 2] < -1e-3
        t = (self.road_z - self.origin[2]) / rays[down, 2]
        xy = self.origin[:2] + rays[down, :2] * t[:, None]
        return xy[(t > 0) & (xy[:, 0] > 0.5) & (xy[:, 0] < max_x)]


def thin(xy, n=1500):
    """At most n road points, cm precision, for tools/fuse_camera_lanes.py."""
    if len(xy) > n:
        xy = xy[np.linspace(0, len(xy) - 1, n).astype(int)]
    return np.round(xy, 2).tolist()


def boundary_pixels(road_mask, step=4):
    """Outermost pixels per row of the drivable region connected to the bottom centre of the image."""
    h, w = road_mask.shape
    count, labels = cv2.connectedComponents((road_mask > 0).astype(np.uint8))
    seed = labels[h - 5, w // 2 - w // 8:w // 2 + w // 8]
    seed = seed[seed > 0]
    if count < 2 or len(seed) == 0:
        return np.zeros((0, 2))
    region = labels == np.bincount(seed).argmax()
    pts = []
    for v in range(h // 3, h, step):
        cols = np.flatnonzero(region[v])
        if len(cols) < 2:
            continue
        if cols[0] > 2:
            pts.append((cols[0], v))
        if cols[-1] < w - 3:
            pts.append((cols[-1], v))
    return np.array(pts, float).reshape(-1, 2)


MODEL_W, MODEL_H = 640, 360
BGR = {'camera_line': (255, 150, 40), 'camera_edge': (255, 80, 160), 'lidar_line': (0, 140, 255), 'lidar_edge': (40, 40, 255)}


def crop_box(camera, w, h):
    """(x0, y0, x1, y1) pixels given to the model."""
    if camera not in CROP:
        return 0, 0, w, h
    top, bottom = CROP[camera]
    y0, y1 = int(top * h), int(bottom * h)
    cw = min(w, round((y1 - y0) * MODEL_W / MODEL_H))
    x0 = (w - cw) // 2
    return x0, y0, x0 + cw, y1


def infer_on_gpu(rt, bgr, display_w, box):
    """TwinLiteNet on the image region box, resized on the GPU; (road, lane) masks at 640x360 and a display image."""
    torch = rt.torch
    h, w = bgr.shape[:2]
    dh = round(h * display_w / w)
    x0, y0, x1, y1 = box
    with torch.cuda.stream(rt.stream):
        img = torch.from_numpy(bgr).to('cuda', non_blocking=True).permute(2, 0, 1)[None].float()
        x = torch.nn.functional.interpolate(img[:, :, y0:y1, x0:x1], size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        rt.input_cuda.copy_(x.flip(1) / 255.0)  # official preprocessing: RGB, /255, direct resize
        rt.start_event.record(rt.stream)
        if not rt.context.execute_async_v3(rt.stream.cuda_stream):
            raise RuntimeError('TwinLiteNet TensorRT inference failed')
        rt.end_event.record(rt.stream)
        masks = torch.stack([rt.logits[0].argmax(1)[0], rt.logits[1].argmax(1)[0]]).to(torch.uint8).cpu()
        disp = torch.nn.functional.interpolate(img, size=(dh, display_w), mode='area')[0].permute(1, 2, 0).to(torch.uint8).cpu()
    rt.stream.synchronize()
    return masks[0].numpy(), masks[1].numpy(), disp.numpy(), rt.start_event.elapsed_time(rt.end_event)


def seg_cloud(header, road_xy, lane_xy, z):
    """Segmentation on the road plane as an RGB point cloud: drivable road green, lane pixels blue."""
    xy = np.vstack([road_xy, lane_xy]).astype(np.float32)
    rgb = np.r_[np.full(len(road_xy), (60 << 16) | (200 << 8) | 60, np.uint32), np.full(len(lane_xy), (40 << 16) | (140 << 8) | 255, np.uint32)]
    data = np.zeros(len(xy), [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    data['x'], data['y'], data['z'], data['rgb'] = xy[:, 0], xy[:, 1], z + 0.02, rgb
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32 if n != 'rgb' else PointField.UINT32, count=1)
              for i, n in enumerate(('x', 'y', 'z', 'rgb'))]
    return PointCloud2(header=header, height=1, width=len(data), fields=fields, is_bigendian=False, point_step=16,
                       row_step=16 * len(data), data=data.tobytes(), is_dense=True)


def project_lines(proj, lines, z, scale):
    """Road-plane polylines (os_sensor) -> image polylines at display scale."""
    out = []
    for line in lines:
        p = np.array([[x, y, z] for x, y in line['points']], float)
        pc = p @ proj.cam_from_sensor[:3, :3].T + proj.cam_from_sensor[:3, 3]
        pc = pc[pc[:, 2] > 0.5]
        if len(pc) < 2:
            continue
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), proj.k, proj.d)
        uv = uv.reshape(-1, 2) * scale
        uv = uv[np.isfinite(uv).all(1) & (np.abs(uv) < 1e5).all(1)]
        if len(uv) >= 2:
            out.append(uv.astype(np.int32))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', choices=sorted(CAMERAS), required=True)
    parser.add_argument('--image-topic', help='default /camera/<camera>/image_raw')
    parser.add_argument('--road-z', type=float, default=-1.75, help='road height in os_sensor (m); fixed flat plane')
    parser.add_argument('--annotate-hz', type=float, default=10.0, help='annotated image rate; 0 = off')
    parser.add_argument('--display-width', type=int, default=800)
    parser.add_argument('--max-points', type=int, default=4000, help='lane-mask pixels projected per image')
    parser.add_argument('--engine', type=Path, default=ROOT / 'models/road_lanes_001/twinlitenet_fp16.engine')
    args = parser.parse_args()
    cam = args.camera
    topic, intr, ext = CAMERAS[cam]
    proj = GroundProjector(intr, ext, args.road_z)
    proj.cam_from_sensor = np.linalg.inv(np.r_[np.c_[proj.rot, proj.origin], [[0, 0, 0, 1]]])
    rt = TwinLiteRuntime(args.engine)
    name = f'camera_{cam}'
    rclpy.init()
    node = rclpy.create_node(f'lanes_{name}')
    pub = node.create_publisher(String, f'/camera_lidar_bev/lanes/{name}', 10)
    mpub = node.create_publisher(MarkerArray, f'/camera_lidar_bev/lanes/{name}_markers', 5)
    ppub = node.create_publisher(String, f'/camera_lidar_bev/lanes/{name}/points', 5)
    spub = node.create_publisher(PointCloud2, f'/camera_lidar_bev/lanes/{name}/segmentation', qos_profile_sensor_data)
    ipub = node.create_publisher(Image, f'/camera_lidar_bev/lanes/{name}/image', qos_profile_sensor_data)
    state = {'img': None, 'lidar': None, 'last_annot': 0.0}
    widths = []
    newest_only = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
    # Depth 1: when inference is slower than the camera, process the newest image, not a queue of old ones.
    node.create_subscription(Image, args.image_topic or topic, lambda m: state.__setitem__('img', m), newest_only)
    node.create_subscription(String, '/camera_lidar_bev/lanes/lidar', lambda m: state.__setitem__('lidar', m.data), 5)
    node.get_logger().info(f'{cam} lanes from {args.image_topic or topic} (road z {args.road_z} m) -> /camera_lidar_bev/lanes/{name}')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            m, state['img'] = state['img'], None
            if m is None:
                continue
            t0 = time.perf_counter()
            bgr = image_bgr(m)
            h, w = bgr.shape[:2]
            if (w, h) != proj.size:
                node.get_logger().error(f'image {w}x{h} but calibration {proj.size}', throttle_duration_sec=5.0)
                continue
            box = crop_box(cam, w, h)
            road, lane, disp, gpu_ms = infer_on_gpu(rt, bgr, args.display_width, box)
            sx, sy = (box[2] - box[0]) / MODEL_W, (box[3] - box[1]) / MODEL_H
            vs, us = np.nonzero(lane)
            if len(vs) > args.max_points:  # plenty for line fitting; keeps cluttered scenes fast
                keep = np.linspace(0, len(vs) - 1, args.max_points).astype(int)
                vs, us = vs[keep], us[keep]
            lane_xy = proj.project(np.c_[box[0] + (us + 0.5) * sx - 0.5, box[1] + (vs + 0.5) * sy - 0.5])
            edge_px = boundary_pixels(road, step=2)
            edge_xy = proj.project(np.c_[box[0] + (edge_px[:, 0] + 0.5) * sx - 0.5,
                                         box[1] + (edge_px[:, 1] + 0.5) * sy - 0.5]) if len(edge_px) else np.zeros((0, 2))
            lines = rf.fit_lines(lane_xy, 'lane_line')
            bounds = rf.fit_lines(edge_xy, 'road_boundary', link_m=1.2)
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            text = rf.to_message(stamp, name, lines, bounds, float(np.median(widths)) if widths else 3.5,
                                 (node.get_clock().now().nanoseconds * 1e-9 - stamp) * 1000,
                                 {'compute_ms': round((time.perf_counter() - t0) * 1000, 1), 'gpu_ms': round(gpu_ms, 2),
                                  'road_z': args.road_z, 'lane_points': int(len(lane_xy)), 'boundary_points': int(len(edge_xy))})
            msg = rf.from_message(text)
            if msg['lane_width_m'] and 2.5 < msg['lane_width_m'] < 5.0:
                widths.append(msg['lane_width_m'])
                del widths[:-50]
            pub.publish(String(data=text))
            # Road points for tools/fuse_camera_lanes.py on their own topic, so the main result stays small.
            ppub.publish(String(data=json.dumps({'stamp': stamp, 'compute_ms': msg['compute_ms'], 'road_z': args.road_z,
                                                 'lane_xy': thin(lane_xy), 'edge_xy': thin(edge_xy)})))
            mpub.publish(markers(msg, 'os_sensor', m.header.stamp, CAMERA_STYLE))
            now = time.monotonic()
            if args.annotate_hz > 0 and now - state['last_annot'] >= 1.0 / args.annotate_hz:
                state['last_annot'] = now
                rv, ru = np.nonzero(road[::4, ::4])  # drivable road on the ground, for the LiDAR view
                road_xy = proj.project(np.c_[box[0] + (ru * 4 + 2) * sx, box[1] + (rv * 4 + 2) * sy], max_x=60.0)
                spub.publish(seg_cloud(Header(stamp=m.header.stamp, frame_id='os_sensor'), road_xy, lane_xy, args.road_z))
                dh, dw = disp.shape[:2]
                img = np.ascontiguousarray(disp)
                bx = [round(v * dw / w) for v in box]
                size = (bx[2] - bx[0], bx[3] - bx[1])
                drive = np.zeros((dh, dw), bool)  # segmentation: drivable road green, lane pixels blue
                drive[bx[1]:bx[3], bx[0]:bx[2]] = cv2.resize(road, size, interpolation=cv2.INTER_NEAREST) > 0
                img[drive] = (0.55 * img[drive] + (0, 100, 0)).astype(np.uint8)
                tint = np.zeros((dh, dw), bool)
                tint[bx[1]:bx[3], bx[0]:bx[2]] = cv2.resize(lane, size, interpolation=cv2.INTER_NEAREST) > 0
                if box != (0, 0, w, h):
                    cv2.rectangle(img, (bx[0], bx[1]), (bx[2] - 1, bx[3] - 1), (200, 200, 200), 1)
                img[tint] = (0.4 * img[tint] + (153, 90, 24)).astype(np.uint8)
                scale = dw / w
                for poly in project_lines(proj, msg['boundaries'], args.road_z, scale):
                    cv2.polylines(img, [poly], False, BGR['camera_edge'], 2, cv2.LINE_AA)
                for poly in project_lines(proj, msg['lines'], args.road_z, scale):
                    cv2.polylines(img, [poly], False, BGR['camera_line'], 2, cv2.LINE_AA)
                if state['lidar']:
                    lid = rf.from_message(state['lidar'])
                    lz = lid['road_z'] if lid.get('road_z') is not None else args.road_z
                    for poly in project_lines(proj, lid['boundaries'], lz, scale):
                        cv2.polylines(img, [poly], False, BGR['lidar_edge'], 2, cv2.LINE_AA)
                    for poly in project_lines(proj, lid['lines'], lz, scale):
                        cv2.polylines(img, [poly], False, BGR['lidar_line'], 2, cv2.LINE_AA)
                cv2.putText(img, f"{cam}  road green, lane px blue, camera lines blue/violet, LiDAR orange/red  {msg['compute_ms']:.0f} ms", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                out = Image(header=Header(stamp=m.header.stamp, frame_id=m.header.frame_id), height=dh, width=dw,
                            encoding='bgr8', step=dw * 3, data=img.tobytes())
                ipub.publish(out)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
