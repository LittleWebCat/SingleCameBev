"""Road features from detected points, shared by the camera and LiDAR lane detectors (numpy + scipy, no ROS).

All coordinates are the car's LiDAR frame os_sensor on the road plane: x forward, y left, metres.
  * fit_lines(points): cluster candidate points (lane paint or road-edge points), fit y = a + b x + c x^2 to
    each cluster, merge pieces of the same dashed / broken line, and return polylines sampled every 1 m;
  * ego_lane(lines): the nearest line on each side of the car (lane line, centre line or road boundary);
  * lane_centre(left, right, width): midpoints of the ego lane, or one side offset by the last lane width;
  * lateral_at(line, x): the line's y at look-ahead x (None outside its measured range);
  * to_message / from_message: the JSON published on /camera_lidar_bev/lanes/<method>.
Both detectors use exactly these functions, so the comparison measures the detection, not this logic.
"""
import json
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

LOOK_AHEAD_M = (5.0, 10.0, 20.0, 30.0)


def fit_lines(points, kind, link_m=0.7, min_length_m=3.0, max_x_m=40.0, merge_m=0.4):
    """points: (N, 2) x, y. Returns [{'kind', 'points': [[x, y], ...], 'support': N}] sorted by y at x=5."""
    points = np.asarray(points, float).reshape(-1, 2)
    points = points[(points[:, 0] > 0.5) & (points[:, 0] < max_x_m)]
    if len(points) < 6:
        return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(link_m, output_type='ndarray')
    n = len(points)
    if len(pairs) == 0:
        return []
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    count, label = connected_components(graph, directed=False)
    pieces = []
    for k in range(count):
        p = points[label == k]
        if len(p) < 6 or np.ptp(p[:, 0]) < 1.0:
            continue  # too small, or across the road (a stop line or noise), not along it
        pieces.append(p)
    # Merge pieces of the same line (dashes): their straight fits agree where they meet.
    pieces.sort(key=lambda p: p[:, 0].min())
    merged = []
    for p in pieces:
        for i, m in enumerate(merged):
            fit = np.polyfit(m[:, 0], m[:, 1], 1 if np.ptp(m[:, 0]) < 8 else 2)
            if np.median(np.abs(np.polyval(fit, p[:, 0]) - p[:, 1])) < merge_m + 0.02 * max(0.0, p[:, 0].min() - m[:, 0].max()):
                merged[i] = np.vstack([m, p])
                break
        else:
            merged.append(p)
    lines = []
    for p in merged:
        x0, x1 = p[:, 0].min(), p[:, 0].max()
        if x1 - x0 < min_length_m:
            continue
        fit = np.polyfit(p[:, 0], p[:, 1], 2 if x1 - x0 > 8 else 1)
        if abs(np.polyval(np.polyder(fit), (x0 + x1) / 2)) > 1.0:
            continue  # more than 45 degrees to the car: not a line along the road
        xs = np.arange(math.ceil(x0), math.floor(x1) + 1.0)
        if len(xs) < 2:
            continue
        lines.append({'kind': kind, 'points': [[float(x), round(float(np.polyval(fit, x)), 3)] for x in xs],
                      'support': int(len(p))})
    lines.sort(key=lambda line: lateral_at(line, 5.0, extrapolate=True))
    return lines


def lateral_at(line, x, extrapolate=False):
    pts = np.asarray(line['points'], float)
    if not extrapolate and not pts[0, 0] - 0.5 <= x <= pts[-1, 0] + 0.5:
        return None
    if extrapolate and len(pts) >= 2 and (x < pts[0, 0] or x > pts[-1, 0]):
        end = pts[:2] if x < pts[0, 0] else pts[-2:]
        slope = (end[1, 1] - end[0, 1]) / (end[1, 0] - end[0, 0])
        return float(end[0, 1] + slope * (x - end[0, 0]))
    return float(np.interp(x, pts[:, 0], pts[:, 1]))


def ego_lane(lines, near_x=5.0, max_offset_m=6.0):
    """(left, right): nearest line on each side at near_x (lines starting further out are extrapolated)."""
    left = right = None
    for line in lines:
        if line['points'][0][0] > 15.0:
            continue
        y = lateral_at(line, near_x, extrapolate=True)
        if 0 < y < max_offset_m and (left is None or y < left[0]):
            left = (y, line)
        if -max_offset_m < y <= 0 and (right is None or y > right[0]):
            right = (y, line)
    return (left[1] if left else None), (right[1] if right else None)


def lane_centre(left, right, width_m=3.5, xs=np.arange(1.0, 31.0)):
    """Midpoints where both sides are measured; elsewhere one side offset by half the lane width."""
    out, one_sided = [], False
    for x in xs:
        yl = lateral_at(left, x) if left else None
        yr = lateral_at(right, x) if right else None
        if yl is not None and yr is not None:
            out.append([float(x), round((yl + yr) / 2, 3)])
        elif yl is not None:
            out.append([float(x), round(yl - width_m / 2, 3)])
            one_sided = True
        elif yr is not None:
            out.append([float(x), round(yr + width_m / 2, 3)])
            one_sided = True
    return out, one_sided


def ego_width(left, right, x=10.0):
    yl, yr = (lateral_at(left, x) if left else None), (lateral_at(right, x) if right else None)
    return None if yl is None or yr is None else yl - yr


def to_message(stamp_s, method, lane_lines, boundaries, width_m, latency_ms, extra=None):
    """JSON for /camera_lidar_bev/lanes/<method>; ego lane uses lane lines first, then road boundaries."""
    left, right = ego_lane(lane_lines)
    bl, br = ego_lane(boundaries, max_offset_m=12.0)
    left, right = left or bl, right or br
    centre, one_sided = lane_centre(left, right, width_m)
    msg = {'stamp': stamp_s, 'method': method, 'lines': lane_lines, 'boundaries': boundaries,
           'ego_left': left, 'ego_right': right, 'road_left': bl, 'road_right': br,
           'lane_centre': centre, 'one_sided': one_sided, 'lane_width_m': ego_width(left, right),
           'latency_ms': round(latency_ms, 1)}
    if extra:
        msg.update(extra)
    return json.dumps(msg)


def from_message(text):
    return json.loads(text)


def compare(a, b, xs=LOOK_AHEAD_M):
    """Per feature and look-ahead, b minus a lateral offset (m), or None where either is missing."""
    out = {}
    for feature in ('ego_left', 'ego_right', 'road_left', 'road_right'):
        out[feature] = {}
        for x in xs:
            ya = lateral_at(a[feature], x) if a.get(feature) else None
            yb = lateral_at(b[feature], x) if b.get(feature) else None
            out[feature][x] = None if ya is None or yb is None else yb - ya
    out['lane_centre'] = {}
    for x in xs:
        ca = {p[0]: p[1] for p in a['lane_centre']}
        cb = {p[0]: p[1] for p in b['lane_centre']}
        out['lane_centre'][x] = cb[x] - ca[x] if x in ca and x in cb else None
    return out
