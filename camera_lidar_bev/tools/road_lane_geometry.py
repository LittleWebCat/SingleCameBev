#!/usr/bin/python3
"""Place camera road/lane masks in BEV only where measured LiDAR supports floor.

This is display geometry, not a trained curb detector. ``road_points`` and
``lane_points`` are measured returns. Lines are camera-mask contour pixels
intersected with a locally fitted, measured ground surface and are estimates.
No road plane or lane line is extended across unobserved portions of the scan.
Coordinates throughout are metres in ``os_sensor``.
"""
import math

import cv2
import numpy as np


CELL_M = 1.0
MAX_SLOPE_DEG = 12.0
HEIGHT_TOLERANCE_M = .35
GROUND_DISTANCE_M = .12
SUPPORT_RADIUS_M = 1.25
LATERAL_LIMIT_M = 30.0


def _empty(reason):
    return dict(status='unavailable', reason=reason, ground_valid=False,
                frame='os_sensor', ground_plane=None,
                ground_points=np.empty((0, 3), dtype=np.float32),
                road_points=np.empty((0, 3), dtype=np.float32),
                lane_points=np.empty((0, 3), dtype=np.float32),
                curb_points=np.empty((0, 3), dtype=np.float32),
                lane_lines=[], boundary_lines=[], candidate_lines=[],
                boundary_kind='camera semantic road edge (not curb)',
                curb_kind='LiDAR curb candidates (not verified curbs)',
                line_source='camera contour + locally supported measured ground',
                support_radius_m=SUPPORT_RADIUS_M, metrics={})


def _fit(points):
    if len(points) < 3:
        return None
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    result, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    return result if rank == 3 and np.isfinite(result).all() else None


def _residual(points, plane):
    return points[:, 2] - points[:, :2] @ plane[:2] - plane[2]


def _valid_plane(plane, ground_height):
    return (plane is not None and
            np.linalg.norm(plane[:2]) <= math.tan(math.radians(MAX_SLOPE_DEG)) and
            abs(plane[2] + ground_height) <= HEIGHT_TOLERANCE_M)


def _broad(points):
    if len(points) < 12:
        return False
    span = np.ptp(points[:, :2], axis=0)
    return (span[0] >= 3.0 and span[1] >= 1.5 and
            np.linalg.eigvalsh(np.cov(points[:, :2].T))[0] >= .08)


def _cell_keys(points):
    return np.floor(points[:, :2] / CELL_M).astype(np.int32)


def _distance2(a, b):
    delta = a[:2] - b[:2]
    return float(delta[0] * delta[0] + delta[1] * delta[1])


def _low_seeds(points, ground_height):
    """One robust low return per sufficiently populated patch, equal patch weight."""
    slope_limit = math.tan(math.radians(MAX_SLOPE_DEG))
    candidate = points[np.abs(points[:, 2] + ground_height) <=
                       HEIGHT_TOLERANCE_M + np.linalg.norm(points[:, :2], axis=1) * slope_limit]
    if not len(candidate):
        return np.empty((0, 3))
    keys = _cell_keys(candidate)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    candidate, keys = candidate[order], keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(keys, axis=0), axis=1)), len(keys)]
    seeds = []
    for first, last in zip(starts[:-1], starts[1:]):
        if last - first < 3:
            continue
        patch = candidate[first:last]
        pick = max(0, int(.1 * len(patch)))
        seeds.append(patch[np.argpartition(patch[:, 2], pick)[pick]])
    return np.asarray(seeds, dtype=float).reshape(-1, 3)


def _connected(seeds):
    """Keep independent floor components from borrowing one another's intercept."""
    keys = _cell_keys(seeds)
    lower = keys.min(axis=0)
    image = np.zeros(tuple((keys.max(axis=0) - lower + 3)[::-1]), dtype=np.uint8)
    image[keys[:, 1] - lower[1] + 1, keys[:, 0] - lower[0] + 1] = 1
    _, labels = cv2.connectedComponents(image, connectivity=8)
    groups = labels[keys[:, 1] - lower[1] + 1, keys[:, 0] - lower[0] + 1]
    return [seeds[groups == group] for group in np.unique(groups) if group]


def _ground_plane(seeds, ground_height):
    if len(seeds) < 12:
        return None, np.empty((0, 3))
    random = np.random.default_rng(0x424556)
    proposals = [np.array([0., 0., -ground_height])]
    band = seeds[np.abs(seeds[:, 2] + ground_height) <= HEIGHT_TOLERANCE_M]
    # Height-constrained proposals help find sparse floor among many walls.
    for trial in range(80):
        pool = band if trial < 40 and len(band) >= 3 else seeds
        a, b = pool[random.choice(len(pool), 2, replace=False)]
        design = np.vstack((a[:2], b[:2]))
        if abs(np.linalg.det(design)) < .5:
            continue
        intercept = -ground_height + random.choice([-.175, 0., .175])
        gradient = np.linalg.solve(design, np.array([a[2], b[2]]) - intercept)
        proposals.append(np.r_[gradient, intercept])
    for _ in range(48):
        fit = _fit(seeds[random.choice(len(seeds), 3, replace=False)])
        if fit is not None:
            proposals.append(fit)
    ranked = []
    for plane in proposals:
        if _valid_plane(plane, ground_height):
            support = np.abs(_residual(seeds, plane)) <= GROUND_DISTANCE_M
            if support.sum() >= 12:
                ranked.append((int(support.sum()), plane, support))
    ranked.sort(key=lambda row: row[0], reverse=True)
    winner, winning, error = None, np.empty((0, 3)), float('inf')
    # Independent, spatially broad floor support is required after RANSAC.
    for _, proposal, support in ranked[:8]:
        for component in _connected(seeds[support]):
            if not _broad(component):
                continue
            refined = _fit(component)
            if not _valid_plane(refined, ground_height):
                continue
            clean = component[np.abs(_residual(component, refined)) <= GROUND_DISTANCE_M]
            if not _broad(clean):
                continue
            refined = _fit(clean)
            candidate_error = float(np.mean(_residual(clean, refined) ** 2))
            if (_valid_plane(refined, ground_height) and
                    (len(clean) > len(winning) or
                     (len(clean) == len(winning) and candidate_error < error))):
                winner, winning, error = refined, clean, candidate_error
    return winner, winning


def _local_models(seeds, plane):
    keys = _cell_keys(seeds)
    lookup = {tuple(key): point for key, point in zip(keys, seeds)}
    local = {}
    for key, point in lookup.items():
        neighbors = np.asarray([lookup[(key[0] + dx, key[1] + dy)]
                                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                if (key[0] + dx, key[1] + dy) in lookup])
        fit = _fit(neighbors) if len(neighbors) >= 5 else None
        if (fit is not None and np.linalg.norm(fit[:2]) <= math.tan(math.radians(MAX_SLOPE_DEG)) and
                np.max(np.abs(_residual(neighbors, fit))) <= GROUND_DISTANCE_M):
            local[key] = fit
        else:
            # Same measured slope plus local observed height, never a default floor.
            local[key] = np.r_[plane[:2], point[2] - point[:2] @ plane[:2]]
    return lookup, local


def _support_grid(lookup, local):
    keys = np.asarray(list(lookup), dtype=int)
    lower = keys.min(axis=0)
    shape = tuple(keys.max(axis=0) - lower + 1)
    seeds = np.full((*shape, 3), np.nan)
    models = np.full((*shape, 3), np.nan)
    for key, point in lookup.items():
        cell = tuple(np.array(key) - lower)
        seeds[cell], models[cell] = point, local[key]
    return lower, seeds, models


def _nearest_models(points, grid):
    """Nine vectorized grid lookups, retaining distance to actual measured seed."""
    lower, seeds, models = grid
    finite = np.isfinite(points[:, :2]).all(axis=1)
    safe = np.where(finite[:, None], points[:, :2], 0.)
    keys = np.floor(safe / CELL_M).astype(int) - lower
    best_distance = np.full(len(points), np.inf)
    best_seeds = np.full((len(points), 3), np.nan)
    best_models = np.full((len(points), 3), np.nan)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cells = keys + [dx, dy]
            valid = finite & (cells[:, 0] >= 0) & (cells[:, 0] < seeds.shape[0])
            valid &= (cells[:, 1] >= 0) & (cells[:, 1] < seeds.shape[1])
            indices = np.flatnonzero(valid)
            observed = seeds[cells[indices, 0], cells[indices, 1]]
            distances = np.sum((observed[:, :2] - points[indices, :2]) ** 2, axis=1)
            better = distances < best_distance[indices]
            chosen = indices[better]
            best_distance[chosen] = distances[better]
            best_seeds[chosen] = observed[better]
            best_models[chosen] = models[cells[chosen, 0], cells[chosen, 1]]
    return best_models, best_seeds, best_distance <= SUPPORT_RADIUS_M ** 2


def _supported_rays(pixels, calibration, plane, lookup, local, max_forward, grid):
    """Undistort native mask pixels, intersect local ground, retain observed support."""
    if not len(pixels):
        return np.empty((0, 3)), np.zeros(0, dtype=bool)
    k = np.asarray(calibration['K'], dtype=float).reshape(3, 3)
    d = np.asarray(calibration['D'], dtype=float)
    transform = np.asarray(calibration['T'], dtype=float).reshape(4, 4)
    inverse = np.linalg.inv(transform)
    normalized = cv2.undistortPoints(np.asarray(pixels, dtype=float).reshape(-1, 1, 2), k, d).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
    rays = rays_camera @ inverse[:3, :3].T
    origin = inverse[:3, 3]
    normal = np.r_[-plane[:2], 1.]
    denominator = rays @ normal
    scale = np.full(len(pixels), np.nan)
    good = np.isfinite(rays).all(axis=1) & (np.abs(denominator) > 1e-5)
    scale[good] = (plane[2] - origin @ normal) / denominator[good]
    result = origin + rays * scale[:, None]
    valid = good & np.isfinite(result).all(axis=1) & (scale > .3)
    valid &= (result[:, 0] >= 0) & (result[:, 0] <= max_forward) & (np.abs(result[:, 1]) <= LATERAL_LIMIT_M)
    fit, nearest, supported = _nearest_models(result, grid)
    normals = np.column_stack((-fit[:, :2], np.ones(len(fit))))
    divisor = np.sum(rays * normals, axis=1)
    distance = np.full(len(pixels), np.nan)
    valid &= supported & np.isfinite(divisor) & (np.abs(divisor) > 1e-5)
    distance[valid] = (fit[valid, 2] - normals[valid] @ origin) / divisor[valid]
    result = origin + distance[:, None] * rays
    valid &= (distance > .3) & np.isfinite(result).all(axis=1)
    valid &= (result[:, 0] >= 0) & (result[:, 0] <= max_forward) & (np.abs(result[:, 1]) <= LATERAL_LIMIT_M)
    valid &= np.sum((result[:, :2] - nearest[:, :2]) ** 2, axis=1) <= SUPPORT_RADIUS_M ** 2
    # Round-trip check rejects undistortion outside the stable native camera model.
    from combined_bev_display import project_camera
    camera = result @ transform[:3, :3].T + transform[:3, 3]
    reprojection, stable = project_camera(camera, calibration)
    valid &= stable & (np.linalg.norm(reprojection - pixels, axis=1) < 1.)
    return result, valid


def _contour_lines(mask, calibration, plane, lookup, local, max_forward, grid, road_boundary=False):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = []
    height, width = mask.shape
    for contour in contours:
        pixels = contour[:, 0, :].astype(float)
        if len(pixels) < 8:
            continue
        pixels = pixels[::max(1, int(math.ceil(len(pixels) / 800.)))]
        points, valid = _supported_rays(pixels, calibration, plane, lookup, local, max_forward, grid)
        # Image-frame edges are crop boundaries, not semantic road edges.
        if road_boundary:
            valid &= ((pixels[:, 0] > 2) & (pixels[:, 0] < width - 3) &
                      (pixels[:, 1] > 2) & (pixels[:, 1] < height - 3))
        run = []
        for index in range(len(points)):
            if not valid[index] or (run and np.linalg.norm(points[index, :2] - run[-1][:2]) > 2.):
                if len(run) >= 2:
                    lines.append(np.asarray(run, dtype=np.float32))
                run = []
            if valid[index]:
                run.append(points[index])
        if len(run) >= 2:
            lines.append(np.asarray(run, dtype=np.float32))
    return lines


def _point_planes(points, lookup, local):
    """Select measured neighboring patch models with bounded spatial support."""
    # Scalar cell codes avoid sorting structured XY records for every scan.
    raw = _cell_keys(points).astype(np.int64)
    codes, inverse = np.unique(raw[:, 0] * 65536 + raw[:, 1] + 32768, return_inverse=True)
    keys = np.column_stack((codes // 65536, codes % 65536 - 32768))
    models = np.full((len(keys), 3), np.nan)
    support = np.full((len(keys), 2), np.nan)
    direct = np.zeros(len(keys), dtype=bool)
    for index, key in enumerate(keys):
        key = tuple(key)
        if key in local:
            models[index], support[index], direct[index] = local[key], lookup[key][:2], True
            continue
        center = (np.array(key) + .5) * CELL_M
        candidates = [(other, lookup[other]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if (other := (key[0] + dx, key[1] + dy)) in local]
        if candidates:
            nearest, point = min(candidates, key=lambda row: _distance2(row[1], center))
            models[index], support[index] = local[nearest], point[:2]
    supported = np.isfinite(models[inverse]).all(axis=1)
    supported &= np.linalg.norm(points[:, :2] - support[inverse], axis=1) <= SUPPORT_RADIUS_M
    heights = points[:, 2] - np.sum(points[:, :2] * models[inverse, :2], axis=1) - models[inverse, 2]
    return heights, supported, direct[inverse]


def _curb_candidates(points, heights, supported, calibration, road_mask):
    """Small actual height steps alongside camera semantic edges, with base support."""
    base = points[supported & (np.abs(heights) <= .04)]
    raised = points[supported & (heights >= .07) & (heights <= .25)]
    if len(base) < 6 or len(raised) < 6:
        return np.empty((0, 3), dtype=np.float32), []
    # A raised point needs actual floor returns within 0.75m, not a plane alone.
    step_cell = .25
    base_keys = np.floor(base[:, :2] / step_cell).astype(np.int64)
    base_codes = np.unique(base_keys[:, 0] * 65536 + base_keys[:, 1] + 32768)
    keys = np.floor(raised[:, :2] / step_cell).astype(int)
    codes = keys[:, 0].astype(np.int64) * 65536 + keys[:, 1] + 32768
    nearby_base = np.zeros(len(raised), dtype=bool)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx * dx + dy * dy > 4:
                continue
            neighbor = codes + dx * 65536 + dy
            index = np.searchsorted(base_codes, neighbor)
            nearby_base |= base_codes[np.minimum(index, len(base_codes) - 1)] == neighbor
    raised = raised[nearby_base]
    if len(raised) < 6:
        return np.empty((0, 3), dtype=np.float32), []
    road = road_mask.astype(np.uint8)
    # Restrict candidates to visual road-edge context; frame borders are excluded.
    edge = cv2.morphologyEx(road, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = 0
    context = cv2.dilate(edge, np.ones((11, 11), np.uint8))
    transform = np.asarray(calibration['T'], dtype=float).reshape(4, 4)
    from combined_bev_display import project_camera
    pixels, visible = project_camera(raised @ transform[:3, :3].T + transform[:3, 3], calibration)
    visible &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < road.shape[1]) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < road.shape[0]))
    indices = np.flatnonzero(visible)
    uv = pixels[indices].astype(int)
    raised = raised[indices[context[uv[:, 1], uv[:, 0]] > 0]]
    if len(raised) < 6:
        return np.empty((0, 3), dtype=np.float32), []
    keys = np.floor(raised[:, :2] / step_cell).astype(int)
    lower = keys.min(axis=0)
    occupancy = np.zeros(tuple((keys.max(axis=0) - lower + 3)[::-1]), dtype=np.uint8)
    occupancy[keys[:, 1] - lower[1] + 1, keys[:, 0] - lower[0] + 1] = 1
    _, labels = cv2.connectedComponents(cv2.dilate(occupancy, np.ones((3, 3), np.uint8)), connectivity=8)
    groups = labels[keys[:, 1] - lower[1] + 1, keys[:, 0] - lower[0] + 1]
    selected, lines = [], []
    for group in np.unique(groups):
        cluster = raised[groups == group]
        if len(cluster) < 6:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(cluster[:, :2].T))
        if eigenvalues[-1] <= .04 or eigenvalues[0] > .25 * eigenvalues[-1]:
            continue  # Compact obstacle blobs and wide surfaces are not edge evidence.
        position = cluster[:, :2] @ eigenvectors[:, -1]
        if np.ptp(position) < .8:
            continue
        ordered = cluster[np.argsort(position)].astype(np.float32)
        breaks = np.r_[0, 1 + np.flatnonzero(np.linalg.norm(np.diff(ordered[:, :2], axis=0), axis=1) > .75), len(ordered)]
        for first, last in zip(breaks[:-1], breaks[1:]):
            run = ordered[first:last]
            if len(run) >= 6 and np.linalg.norm(run[-1, :2] - run[0, :2]) >= .8:
                selected.append(run)
                lines.append(run)
    return (np.concatenate(selected) if selected else np.empty((0, 3), dtype=np.float32)), lines


def road_lane_bev(points_xyzi_os_sensor, road_mask, lane_mask, calibration,
                  max_forward=60., ground_height=1.8):
    """Return road/lane measurements and ground-supported semantic contour lines.

    Native-size masks and sensor XYZ/XYZI points are required. A rig-height prior
    constrains *fitted measured* ground; it never supplies unobserved ground.
    Unmatched/out-of-FOV points and unsupported contour pieces are omitted.
    """
    shape = (int(calibration['height']), int(calibration['width']))
    road_mask, lane_mask = np.asarray(road_mask), np.asarray(lane_mask)
    if road_mask.shape != shape or lane_mask.shape != shape:
        raise ValueError('Road/lane masks must have native calibration image dimensions')
    if not math.isfinite(ground_height) or ground_height <= 0 or max_forward <= 0:
        raise ValueError('Ground height and forward range must be positive')
    points = np.asarray(points_xyzi_os_sensor, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('Expected Nx3 or Nx4 os_sensor points')
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1) &
                    (points[:, 0] >= 0) & (points[:, 0] <= max_forward) &
                    (np.abs(points[:, 1]) <= LATERAL_LIMIT_M) &
                    (np.linalg.norm(points[:, :2], axis=1) >= 2.5)]
    if len(points) < 40:
        return _empty('Insufficient finite LiDAR returns for measured ground')
    seeds = _low_seeds(points, ground_height)
    plane, support = _ground_plane(seeds, ground_height)
    if plane is None:
        return _empty('No broad measured ground component passed height and slope gates')
    lookup, local = _local_models(support, plane)
    grid = _support_grid(lookup, local)
    heights, supported, direct = _point_planes(points, lookup, local)
    floor = supported & direct & (np.abs(heights) <= GROUND_DISTANCE_M)
    ground = points[floor]
    transform = np.asarray(calibration['T'], dtype=float).reshape(4, 4)
    from combined_bev_display import project_camera
    pixels, visible = project_camera(ground @ transform[:3, :3].T + transform[:3, 3], calibration)
    visible &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < shape[1]) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < shape[0]))
    road_hit = np.zeros(len(ground), dtype=bool)
    lane_hit = np.zeros(len(ground), dtype=bool)
    indices = np.flatnonzero(visible)
    uv = pixels[indices].astype(int)
    road_hit[indices] = road_mask[uv[:, 1], uv[:, 0]] > 0
    lane_hit[indices] = lane_mask[uv[:, 1], uv[:, 0]] > 0
    road_binary = road_mask > 0
    lane_binary = lane_mask > 0
    curb_points, candidate_lines = _curb_candidates(points, heights, supported, calibration, road_binary)
    result = _empty('Measured LiDAR ground supports camera segmentation')
    result.update(status='supported', ground_valid=True, ground_plane=plane.tolist(),
                  ground_points=ground.astype(np.float32),
                  road_points=ground[road_hit].astype(np.float32),
                  lane_points=ground[lane_hit].astype(np.float32),
                  curb_points=curb_points, candidate_lines=candidate_lines,
                  lane_lines=_contour_lines(lane_binary, calibration, plane, lookup, local, max_forward, grid),
                  boundary_lines=_contour_lines(road_binary, calibration, plane, lookup, local, max_forward, grid, True),
                  metrics=dict(ground_returns=len(ground), road_returns=int(road_hit.sum()),
                               lane_returns=int(lane_hit.sum()), ground_patches=len(support),
                               curb_candidate_returns=len(curb_points),
                               slope_deg=float(math.degrees(math.atan(np.linalg.norm(plane[:2])))),
                               intercept_m=float(plane[2]), max_forward_m=float(max_forward)))
    return result
