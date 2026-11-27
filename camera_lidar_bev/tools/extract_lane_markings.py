#!/usr/bin/env python3
"""Extract painted road markings (lane, centre and edge lines, stop bars, crosswalk stripes) from a saved map.

Road paint is retro-reflective, so the Ouster returns more intensity from it than from the asphalt around
it; absolute intensity changes with range and angle, so every point is compared with its local road.
  1. road surface: points within --corridor-m of the driven route and within +-0.12 m of a 1 m road
     height grid (30th-percentile z of points near the route's road height, median filtered)
  2. intensity / median road intensity of the surrounding 3 x 3 m, splatted to a 0.1 m raster
  3. oriented line filter: for 24 directions, a 2.5 m x 0.3 m bar minus its side bands at +-0.6 m
     (log contrast); the best direction, divided by the local noise level (75th percentile per 5 m block)
  4. ridges across the best direction (non-maximum suppression), hysteresis, thinning, traced into
     chains, chains continuing each other within 1 m joined, Douglas-Peucker
  5. type: >= 3 short parallel bars side by side = crosswalk; along the route = line (the nearest line
     on the driver's left of the route is the centre line, the nearest on the right the edge line,
     others lane lines); across the route and 2-7 m long = stop line; short or curly leftovers dropped
Map and route are only read. Writes MAP/lane_markings/:
  lane_markings.json            polylines in enu (z on the road), type, length, mean intensity contrast
  lane_markings_preview.png     top-down: road intensity contrast, route (green), markings by type,
                                kept stop lines of the landmark review (green bars); zooms on the stop signs
  road_intensity_contrast.png   the 0.1 m contrast raster (row 0 = north edge; origin in the json)
tools/lane_markings_publisher.py shows the json in RViz.

  /usr/bin/python3 tools/extract_lane_markings.py [--map maps/local_gps_recording_002]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy import ndimage, spatial
from scipy.sparse.csgraph import connected_components

from saved_map_source import read_pcd_xyz

CELL = 0.1  # raster cell (m)
COLOURS = {'lane_line': (235, 235, 235), 'edge_line': (255, 200, 90), 'centre_line': (40, 220, 255),
           'stop_line': (60, 60, 255), 'crosswalk': (255, 80, 255)}  # BGR, preview only


def cell_percentile(xy, values, lo, cell, shape, q, min_count):
    """Per grid cell q-th percentile of values (NaN where fewer than min_count)."""
    ij = ((xy - lo) / cell).astype(np.int64)
    key = ij[:, 1] * shape[1] + ij[:, 0]
    order = np.lexsort((values, key))
    sk = key[order]
    starts = np.r_[0, np.nonzero(np.diff(sk))[0] + 1]
    counts = np.diff(np.r_[starts, len(order)])
    out = np.full(shape[0] * shape[1], np.nan)
    ok = counts >= min_count
    out[sk[starts[ok]]] = values[order[starts[ok] + (counts[ok] * q).astype(np.int64)]]
    return out.reshape(shape), ij


def fill_nearest(grid):
    return grid[tuple(ndimage.distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True))]


def road_points(points, route, corridor_m, lidar_height_m):
    """Road surface points near the route, their intensity / local road median, and the road height grid."""
    dist, k = spatial.cKDTree(route[:, :2]).query(points[:, :2], distance_upper_bound=corridor_m)
    near = dist < corridor_m
    points, k = points[near], k[near]
    lo = points[:, :2].min(0) - 1.0
    shape = tuple((((points[:, :2].max(0) - lo) / 1.0).astype(int) + 2)[::-1])
    cand = np.abs(points[:, 2] - (route[k, 2] - lidar_height_m)) < 1.0
    height, _ = cell_percentile(points[cand, :2], points[cand, 2], lo, 1.0, shape, 0.3, 4)
    height = ndimage.median_filter(fill_nearest(height), 5)
    ij = ((points[:, :2] - lo) / 1.0).astype(np.int64)
    road = np.abs(points[:, 2] - height[ij[:, 1], ij[:, 0]]) < 0.12
    points = points[road]
    median, ij = cell_percentile(points[:, :2], points[:, 3], lo, 1.0, shape, 0.5, 1)
    pad = np.pad(median, 1, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN 3x3 blocks off the road
        local = np.nanmedian(np.stack([pad[dy:dy + shape[0], dx:dx + shape[1]] for dy in range(3) for dx in range(3)]), 0)
    ratio = points[:, 3] / np.maximum(local[ij[:, 1], ij[:, 0]], 5.0)
    return points, ratio, (height, lo, 1.0)


def contrast_raster(xy, ratio, lo, shape):
    """Gaussian-splatted mean intensity ratio on the 0.1 m grid and the point support behind each cell."""
    ij = ((xy - lo) / CELL).astype(np.int64)
    key = ij[:, 1] * shape[1] + ij[:, 0]
    count = np.bincount(key, minlength=shape[0] * shape[1]).reshape(shape).astype(np.float32)
    total = np.bincount(key, weights=ratio, minlength=shape[0] * shape[1]).reshape(shape).astype(np.float32)
    support = ndimage.gaussian_filter(count, 1.2)
    mean = ndimage.gaussian_filter(total, 1.2) / np.maximum(support, 1e-6)
    return np.where(support > 0.05, mean, 0).astype(np.float32), support


def line_filter(value, valid, length_cells=25, offset_cells=6, half_cells=1, angles=24):
    """Best oriented bar contrast and its direction index (0..angles-1, step 180/angles deg).

    Contrast = bar mean minus the brighter of its two side bands, so both sides must be darker (no step edges)."""
    best = np.full(value.shape, -9, np.float32)
    arg = np.zeros(value.shape, np.uint8)
    n = 2 * max(length_cells // 2, offset_cells + half_cells) + 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    for i in range(angles):
        t = np.pi * i / angles
        u, w = xx * np.cos(t) + yy * np.sin(t), -xx * np.sin(t) + yy * np.cos(t)
        along = np.abs(u) <= length_cells / 2
        means = []
        for band in (np.abs(w) <= half_cells, np.abs(w - offset_cells) <= half_cells, np.abs(w + offset_cells) <= half_cells):
            k = (along & band).astype(np.float32)
            support = cv2.filter2D(valid, -1, k)
            means.append(np.where(support > 0.5 * k.sum(), cv2.filter2D(value, -1, k) / np.maximum(support, 1e-3), np.nan))
        r = means[0] - np.maximum(means[1], means[2])  # NaN (no data) on either side: no line
        r = np.where(np.isfinite(r), r, -9)
        better = r > best
        best[better], arg[better] = r[better], i
    return best, arg


def noise_level(best, block=50):
    """75th percentile of the filter response per 5 m block, smoothly interpolated."""
    h, w = best.shape
    grid = np.full(((h + block - 1) // block, (w + block - 1) // block), np.nan, np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            b = best[r * block:(r + 1) * block, c * block:(c + 1) * block]
            b = b[b > -9]
            if b.size > block * block // 5:
                grid[r, c] = np.percentile(b, 75)
    grid = ndimage.median_filter(fill_nearest(grid), 3) if np.isfinite(grid).any() else np.ones_like(grid)
    return cv2.resize(grid, (grid.shape[1] * block, grid.shape[0] * block), interpolation=cv2.INTER_LINEAR)[:h, :w]


def ridges(score, arg, angles, high, low):
    t = np.pi * arg / angles
    nx, ny = np.rint(-np.sin(t)).astype(int), np.rint(np.cos(t)).astype(int)
    h, w = score.shape
    yy, xx = np.mgrid[:h, :w]
    side = lambda s: score[np.clip(yy + s * ny, 0, h - 1), np.clip(xx + s * nx, 0, w - 1)]
    ridge = (score >= side(1)) & (score >= side(-1)) & (score > low)
    lab, n = ndimage.label(ridge, np.ones((3, 3)))
    strong = np.bincount(lab[ridge & (score > high)], minlength=n + 1)
    keep = strong >= 3
    keep[0] = False
    thin = cv2.ximgproc.thinning(keep[lab].astype(np.uint8) * 255) > 0
    # Cut at junctions (>= 3 branches around the 8-neighbour ring) so every piece is a simple path.
    p = np.pad(thin, 1).astype(np.int8)
    ring = [p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2], p[1:-1, 0:-2], p[0:-2, 0:-2]]
    crossings = sum((ring[(i + 1) % 8] - ring[i]) == 1 for i in range(8))
    return thin & ~(crossings >= 3)


def trace(mask):
    """Ordered pixel chains (row, col) of a thinned, junction-free mask."""
    lab, n = ndimage.label(mask, np.ones((3, 3)))
    steps = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    chains = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        pix = set(map(tuple, np.argwhere(lab[sl] == i + 1) + [sl[0].start, sl[1].start]))
        while pix:
            ends = [q for q in pix if sum((q[0] + a, q[1] + b) in pix for a, b in steps) <= 1]
            cur = ends[0] if ends else next(iter(pix))
            chain = [cur]
            pix.discard(cur)
            while True:
                nxt = next(((cur[0] + a, cur[1] + b) for a, b in steps if (cur[0] + a, cur[1] + b) in pix), None)
                if nxt is None:
                    break
                chain.append(nxt)
                pix.discard(nxt)
                cur = nxt
            chains.append(np.array(chain))
    return chains


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def join(chains, max_gap_m=1.0, max_angle_deg=30):
    """Join chains whose ends continue each other (cut junctions, small breaks). chains: lists of xy arrays."""
    cos = np.cos(np.deg2rad(max_angle_deg))
    ends, out_dirs = [], []
    for c in chains:
        k = min(len(c) - 1, 10)
        ends += [c[0], c[-1]]
        out_dirs += [unit(c[0] - c[k]), unit(c[-1] - c[-1 - k])]
    ends, out_dirs = np.array(ends), np.array(out_dirs)
    pairs = []
    for a, b in spatial.cKDTree(ends).query_pairs(max_gap_m, output_type='ndarray'):
        if a // 2 == b // 2 or out_dirs[a] @ -out_dirs[b] < cos:
            continue
        gap = ends[b] - ends[a]
        g = np.linalg.norm(gap)
        if g > 0.2 and (unit(gap) @ out_dirs[a] < cos or unit(gap) @ -out_dirs[b] < cos):
            continue
        pairs.append((g, a, b))
    link = {}
    for _, a, b in sorted(pairs):
        if a not in link and b not in link:
            link[a], link[b] = b, a
    seen, joined = set(), []
    for start in range(len(chains)):
        if start in seen:
            continue
        # Walk back to one free end of this group, then forward through the links.
        c, side = start, 0
        while 2 * c + side in link:
            other = link[2 * c + side]
            c, side = other // 2, 1 - other % 2
            if c == start:
                break
        pieces, entry = [], side
        while c not in seen:
            seen.add(c)
            pieces.append(chains[c] if entry == 0 else chains[c][::-1])
            exit_end = 2 * c + (1 - entry)
            if exit_end not in link:
                break
            nxt = link[exit_end]
            c, entry = nxt // 2, nxt % 2
        joined.append(np.vstack(pieces))
    return joined


def route_frame(route_xy, xy):
    """Nearest route tangent and signed lateral offset (left of travel positive) for each xy."""
    tangents = np.gradient(route_xy, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    dist, k = spatial.cKDTree(route_xy).query(xy)
    d = xy - route_xy[k]
    return tangents[k], tangents[k, 0] * d[:, 1] - tangents[k, 1] * d[:, 0], dist


def crosswalk_stripes(contrast, lo, thresholds, min_stripes=4, min_contrast=2.7):
    """Crosswalks: rows of short bright bars (0.6-4 m x < 0.9 m), parallel and side by side 0.5-3 m apart.

    Bars are blobs of the smoothed log contrast above any of the thresholds (higher ones split stripes that
    merge at lower ones). A row needs >= min_stripes bars of mean intensity ratio >= min_contrast (the
    asphalt's ring-shaped scan texture makes short rows of weak bars); it then takes in every parallel bar
    continuing it. Returns stripe features (2-point centre line along each bar), their raster mask, the
    crosswalk outlines (4 corners each) and long bars (2.5-8 m x < 0.9 m, ratio >= 2.6: stop line candidates)."""
    v = cv2.GaussianBlur(np.where(contrast > 0, np.log(np.maximum(contrast, 0.2)), 0).astype(np.float32), (0, 0), 1.5)
    rows, cols = np.indices(v.shape, dtype=np.float32)
    bars, blobs, long = [], [], []  # long: 2.5-8 m bars (stop line candidates); blobs: (labels, label, centre, direction, length, contrast, covariance, width); bars seed rows
    for threshold in thresholds:
        lab, n = ndimage.label(v > threshold)
        idx = np.arange(1, n + 1)
        area = ndimage.sum_labels(np.ones_like(v), lab, idx)
        mean_contrast = ndimage.mean(contrast, lab, idx)
        mx, my = ndimage.mean(cols, lab, idx), ndimage.mean(rows, lab, idx)
        sxx = ndimage.mean(cols * cols, lab, idx) - mx * mx
        syy = ndimage.mean(rows * rows, lab, idx) - my * my
        sxy = ndimage.mean(cols * rows, lab, idx) - mx * my
        a, b, c = sxx * CELL ** 2, sxy * CELL ** 2, syy * CELL ** 2
        root = np.sqrt(((a - c) / 2) ** 2 + b ** 2)
        length, width = np.sqrt(12 * ((a + c) / 2 + root)), np.sqrt(12 * np.maximum((a + c) / 2 - root, 0))
        angle = 0.5 * np.arctan2(2 * b, a - c)
        centre = lo + (np.stack([mx, my], 1) + 0.5) * CELL
        keep = (area >= 8) & (length <= 8.0) & (width <= 1.0)
        if bars:  # the same bar at a lower threshold wins
            keep &= spatial.cKDTree([blobs[q][2] for q in bars]).query(centre)[0] >= 0.4
        long_bar = keep & (length >= 2.5) & (width <= 0.9) & (length >= 4 * width) & (mean_contrast >= 2.6)
        if long:
            long_bar &= spatial.cKDTree([q['xy'].mean(0) for q in long]).query(centre)[0] >= 0.5
        long += [{'xy': centre[i] + np.outer([-0.5, 0.5], [np.cos(angle[i]), np.sin(angle[i])]) * length[i],
                  'length': float(length[i]), 'contrast': float(mean_contrast[i]), 'score': 0.0, 'label': (lab, i + 1)}
                 for i in np.nonzero(long_bar)[0]]
        keep &= length <= 4.0
        for i in np.nonzero(keep)[0]:
            blobs.append((lab, i + 1, centre[i], np.array([np.cos(angle[i]), np.sin(angle[i])]), length[i],
                          float(mean_contrast[i]), np.array([[a[i], b[i]], [b[i], c[i]]]), width[i]))
            if area[i] >= 12 and length[i] >= 0.6 and width[i] <= 0.9 and length[i] >= 1.8 * width[i]:
                bars.append(len(blobs) - 1)
    if len(bars) < min_stripes:
        return [], np.zeros(v.shape, bool), [], long
    centres = np.array([b[2] for b in blobs])
    direction = np.array([b[3] for b in blobs])
    elongated = np.array([q[4] >= 1.3 * q[7] for q in blobs])
    graph = np.zeros((len(bars), len(bars)), bool)
    for a, b in spatial.cKDTree(centres[bars]).query_pairs(3.0):
        gap = centres[bars[b]] - centres[bars[a]]
        if (abs(direction[bars[a]] @ direction[bars[b]]) > np.cos(np.deg2rad(20)) and np.linalg.norm(gap) > 0.5
                and abs(gap @ direction[bars[a]]) < 0.7):
            graph[a, b] = graph[b, a] = True
    groups, members = connected_components(graph, directed=False)
    tree, taken, rows_out = spatial.cKDTree(centres), set(), []
    for g in range(groups):
        row = [bars[i] for i in np.nonzero(members == g)[0] if bars[i] not in taken]
        if len(row) < min_stripes or np.mean([blobs[i][5] for i in row]) < min_contrast:
            continue
        while True:  # grow along the row, one blob at a time: bright blobs on its line up to 3 m beyond its ends
            u = unit(sum(direction[i] * np.sign(direction[i] @ direction[row[0]]) for i in row))
            r, c0 = np.array([-u[1], u[0]]), centres[row].mean(0)
            span = (centres[row] - c0) @ r
            best = None
            for j in tree.query_ball_point(c0, np.abs(span).max() + 3.5):
                along = (centres[j] - c0) @ r
                gap = np.abs(along - span).min()
                if (j not in taken and j not in row and abs((centres[j] - c0) @ u) < 0.7 and gap > 0.5
                        and span.min() - 3 < along < span.max() + 3
                        and (not elongated[j] or abs(direction[j] @ u) > np.cos(np.deg2rad(20)))
                        and (best is None or gap < best[0])):
                    best = (gap, j)
            if best is None:
                break
            row.append(best[1])
        taken.update(row)
        rows_out.append([blobs[i] for i in row])
    stripes, mask, outlines = [], np.zeros(v.shape, bool), []
    for g, row in enumerate(rows_out):
        u = unit(sum(b[3] * np.sign(b[3] @ row[0][3]) for b in row))
        r, c0 = np.array([-u[1], u[0]]), np.mean([b[2] for b in row], 0)
        along = np.array([(b[2] - c0) @ r for b in row])
        lengths = [np.sqrt(12 * u @ b[6] @ u) for b in row]  # each blob's extent along the stripe direction
        half = np.median(lengths) / 2
        outlines.append([c0 + r * along.min() - u * half, c0 + r * along.max() - u * half,
                         c0 + r * along.max() + u * half, c0 + r * along.min() + u * half])
        for (lab, label, centre, _, _, value, _, _), length in zip(row, lengths):
            mask |= lab == label
            stripes.append({'xy': np.array([centre - u * length / 2, centre + u * length / 2]), 'length': float(length),
                            'contrast': value, 'score': 0.0, 'type': 'crosswalk', 'group': g, 'across': 0.0})
    return stripes, mask, outlines, long


def supported(f, feats, solid_m=5.0, dashed_m=6.0, strong=(2.5, 3.0)):
    """Keep a line that is long, strong, or one dash of a collinear dashed pattern totalling >= dashed_m."""
    if f['length'] >= solid_m or (f['length'] >= strong[0] and f['contrast'] >= strong[1]):
        return True
    total = f['length']
    for g in feats:
        if g is f or g['type'] != 'lane_line' or abs(f['direction'] @ g['direction']) < np.cos(np.deg2rad(10)):
            continue
        gap = g['centre'] - f['centre']
        normal = np.array([-f['direction'][1], f['direction'][0]])
        if abs(gap @ normal) < 0.4 and abs(gap @ f['direction']) < 15 + (f['length'] + g['length']) / 2:
            total += g['length']
    return total >= dashed_m


def segment_distance(p, ab):
    a, b = ab
    return np.linalg.norm(p - a - np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1) * (b - a))


def stop_bars(bars, route_xy, crosswalk_mask, max_offset_m=7.0, max_tilt_deg=25):
    """Long bright bars across the route heading (within max_tilt_deg of perpendicular) on its road."""
    if not bars:
        return []
    centres = np.array([b['xy'].mean(0) for b in bars])
    tangent, offset, dist = route_frame(route_xy, centres)
    out = []
    for k in np.argsort([-b['length'] for b in bars]):  # longest first; bars inside a kept one are duplicates
        b, t, o = bars[k], tangent[k], offset[k]
        d = unit(b['xy'][1] - b['xy'][0])
        lab, label = b.pop('label')
        if (abs(t[0] * d[1] - t[1] * d[0]) >= np.cos(np.deg2rad(max_tilt_deg)) and abs(o) < max_offset_m
                and not crosswalk_mask[lab == label].mean() > 0.5
                and all(segment_distance(centres[k], s['xy']) > 0.6 for s in out)):
            out.append(dict(b, type='stop_line', across=1.0, offset=float(o)))
    return out


def classify(feats):
    for f in feats:
        if f['across'] < np.sin(np.deg2rad(25)):
            f['type'] = 'lane_line' if f['length'] >= 2.0 and f['straight'] > 0.7 else None
        elif f['length'] >= 5.0 and f['straight'] > 0.8:
            f['type'] = 'lane_line'  # crossing-road or turning lines
        else:
            f['type'] = None


def centre_edge_votes(feats, route_xy, lo, shape):
    """Per route metre, the nearest lane line left of the route votes centre, the nearest right votes edge."""
    ids = np.full(shape, -1, np.int32)
    for i, f in enumerate(feats):
        if f['type'] == 'lane_line' and f['across'] < 0.5:
            cv2.polylines(ids, [np.round((f['xy'] - lo) / CELL).astype(np.int32).reshape(-1, 1, 2)], False, i, 3)
    step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1))]
    samples = np.interp(np.arange(0, step[-1], 1.0), step, np.arange(len(route_xy)))
    xy = np.stack([np.interp(samples, np.arange(len(route_xy)), route_xy[:, a]) for a in (0, 1)], 1)
    t = np.gradient(xy, axis=0)
    left = np.stack([-t[:, 1], t[:, 0]], 1) / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    offsets = np.arange(0.3, 8.0, 0.05)
    votes = {i: [0, 0, 0] for i in range(len(feats))}  # centre, edge, crossings
    sides = {'left': 0, 'right': 0}
    for p, n in zip(xy, left):
        for sign, slot in ((1, 0), (-1, 1)):
            q = np.round((p + sign * offsets[:, None] * n - lo) / CELL).astype(int)
            ok = (q >= 0).all(1) & (q[:, 0] < shape[1]) & (q[:, 1] < shape[0])
            hit = ids[q[ok, 1], q[ok, 0]]
            hit = hit[hit >= 0]
            if hit.size:
                votes[int(hit[0])][slot] += 1
                sides['left' if sign > 0 else 'right'] += 1
            for i in np.unique(hit):
                votes[int(i)][2] += 1
    for i, (c, e, n) in votes.items():
        if feats[i]['type'] == 'lane_line' and n:
            if c >= 0.5 * n and c >= 2:
                feats[i]['type'] = 'centre_line'
            elif e >= 0.5 * n and e >= 2:
                feats[i]['type'] = 'edge_line'
    return len(xy), sides


def preview(path, contrast, lo, route_xy, feats, crosswalks, stop_lines, zooms):
    shape = contrast.shape
    base = (np.clip((contrast - 0.8) / 2.5, 0, 1) * 170).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    px = lambda xy: np.round((np.asarray(xy) - lo) / CELL).astype(np.int32).reshape(-1, 1, 2)
    big = img.copy()
    cv2.polylines(big, [px(route_xy)], False, (60, 200, 60), 5)
    for f in feats:
        cv2.polylines(big, [px(f['xy'])], False, COLOURS[f['type']], 5 if f['type'] in ('stop_line', 'crosswalk') else 4)
    cv2.polylines(big, [px(c) for c in crosswalks], True, COLOURS['crosswalk'], 3)
    for a, b in stop_lines:
        cv2.line(big, tuple(px(a)[0, 0]), tuple(px(b)[0, 0]), (0, 255, 0), 3)
    full = cv2.resize(big[::-1], None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    y = 30
    for name, colour in list(COLOURS.items()) + [('route', (60, 200, 60)), ('landmark stop line', (0, 255, 0))]:
        cv2.putText(full, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        y += 24
    panels = []
    for cx, cy in zooms:  # 40 m squares at 0.05 m/px, markings drawn thin over the contrast raster
        c0, r0 = ((np.array([cx, cy]) - 20 - lo) / CELL).astype(int)
        crop = np.zeros((400, 400, 3), np.uint8)
        rs, cs = slice(max(r0, 0), min(r0 + 400, shape[0])), slice(max(c0, 0), min(c0 + 400, shape[1]))
        if rs.start < rs.stop and cs.start < cs.stop:
            crop[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] = img[rs, cs]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        loc = lambda xy: np.round((np.asarray(xy) - [cx - 20, cy - 20]) / 0.05).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(crop, [loc(route_xy)], False, (60, 200, 60), 2)
        for f in feats:
            cv2.polylines(crop, [loc(f['xy'])], False, COLOURS[f['type']], 2)
        cv2.polylines(crop, [loc(c) for c in crosswalks], True, COLOURS['crosswalk'], 1)
        for a, b in stop_lines:
            cv2.line(crop, tuple(loc(a)[0, 0]), tuple(loc(b)[0, 0]), (0, 255, 0), 2)
        crop = crop[::-1].copy()
        cv2.putText(crop, f'enu ({cx:.0f}, {cy:.0f}) 40 x 40 m', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(crop)
    if panels:
        side = np.vstack(panels)
        h = max(full.shape[0], side.shape[0])
        pad = lambda im: np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)])
        full = np.hstack([pad(full), np.full((h, 6, 3), 255, np.uint8), pad(side)])
    cv2.imwrite(str(path), full)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=root / 'maps/local_gps_recording_002')
    parser.add_argument('--corridor-m', type=float, default=15.0, help='use road within this distance of the route')
    parser.add_argument('--lidar-height-m', type=float, default=1.95, help='route (LiDAR) height above the road')
    parser.add_argument('--high', type=float, default=2.6, help='ridge seed threshold (x local noise level)')
    parser.add_argument('--stripe-threshold', type=float, default=0.5, help='lowest crosswalk bar log intensity contrast (also tried +0.2, +0.4)')
    parser.add_argument('--low', type=float, default=1.9, help='ridge continuation threshold (x local noise level)')
    args = parser.parse_args()
    source = args.map / 'gps_alignment/pointcloud_map_enu.pcd'
    n = len(read_pcd_xyz(source))  # validates the x y z intensity float32 layout
    points = np.asarray(np.memmap(source, mode='r', dtype='<f4', offset=source.stat().st_size - n * 16, shape=(n, 4)), np.float64)
    traj = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    route = np.stack([traj['x_m'], traj['y_m'], traj['z_m']], 1)
    out = args.map / 'lane_markings'
    out.mkdir(exist_ok=True)

    road, ratio, (height, height_lo, height_cell) = road_points(points, route, args.corridor_m, args.lidar_height_m)
    lo = road[:, :2].min(0) - 1.0
    shape = tuple((((road[:, :2].max(0) + 1.0 - lo) / CELL).astype(int) + 1)[::-1])
    contrast, support = contrast_raster(road[:, :2], ratio, lo, shape)
    valid = (support > 0.08).astype(np.float32)
    angles = 24
    best, arg = line_filter(np.where(valid > 0, np.log(np.maximum(contrast, 0.2)), 0).astype(np.float32), valid, angles=angles)
    score = np.where(best > -9, best / np.maximum(noise_level(best), 0.2), 0)
    chains = trace(ridges(score, arg, angles, args.high, args.low))
    xy_chains = [lo + (c[:, ::-1] + 0.5) * CELL for c in chains if len(c) >= 5]
    xy_chains = join(xy_chains)

    feats = []
    for xy in xy_chains:
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = seg.sum()
        if length < 1.0:
            continue
        centred = xy - xy.mean(0)
        direction = np.linalg.eigh(centred.T @ centred)[1][:, -1]
        rc = np.clip(((xy - lo) / CELL).astype(int), 0, [shape[1] - 1, shape[0] - 1])
        feats.append({'xy': xy, 'rc': rc, 'length': float(length), 'straight': float(np.linalg.norm(xy[-1] - xy[0]) / length),
                      'centre': xy.mean(0), 'direction': direction,
                      'contrast': float(contrast[rc[:, 1], rc[:, 0]].mean()), 'score': float(score[rc[:, 1], rc[:, 0]].mean())})
    tangent, offset, _ = route_frame(route[:, :2], np.array([f['centre'] for f in feats]))
    for f, t, o in zip(feats, tangent, offset):
        f['across'], f['offset'] = float(abs(t[0] * f['direction'][1] - t[1] * f['direction'][0])), float(o)
    classify(feats)
    stripes, stripe_mask, crosswalks, long_bars = crosswalk_stripes(
        contrast, lo, (args.stripe_threshold, args.stripe_threshold + 0.2, args.stripe_threshold + 0.4))
    near_stripe = ndimage.binary_dilation(stripe_mask, iterations=3)
    feats = [f for f in feats if f['type'] and near_stripe[f['rc'][:, 1], f['rc'][:, 0]].mean() < 0.5]
    feats = [f for f in feats if f['type'] != 'lane_line' or supported(f, feats)]
    stops = stop_bars(long_bars, route[:, :2], near_stripe)
    on_stop = lambda f: any(np.mean([segment_distance(p, b['xy']) < 0.5 for p in f['xy']]) > 0.5 for b in stops)
    feats = [f for f in feats if not on_stop(f)] + stripes + stops
    route_m, sides = centre_edge_votes(feats, route[:, :2], lo, shape)

    def road_z(xy):
        ij = np.clip(((xy - height_lo) / height_cell).astype(int), 0, [height.shape[1] - 1, height.shape[0] - 1])
        return height[ij[:, 1], ij[:, 0]]

    markings = []
    for f in sorted(feats, key=lambda f: (f['type'], -f['length'])):
        simple = cv2.approxPolyDP(f['xy'].astype(np.float32).reshape(-1, 1, 2), 0.08, False).reshape(-1, 2).astype(float)
        item = {'id': len(markings), 'type': f['type'], 'length_m': round(f['length'], 2),
                'intensity_contrast': round(f['contrast'], 2), 'line_score': round(f['score'], 2),
                'points_enu_m': [[round(x, 3), round(y, 3), round(float(z), 3)] for (x, y), z in zip(simple, road_z(simple))]}
        if 'group' in f:
            item['crosswalk_group'] = f['group']
        markings.append(item)
    reviewed = args.map / 'landmarks/landmarks_reviewed.json'
    stop_lines = [l['stop_line_estimated']['endpoints_enu_m'] for l in json.loads(reviewed.read_text())['landmarks']
                  if l.get('keep') and l.get('stop_line_estimated')] if reviewed.exists() else []
    stop_xy = [(np.array(a[:2]), np.array(b[:2])) for a, b in stop_lines]
    types = sorted(COLOURS)
    summary = {t: {'count': sum(m['type'] == t for m in markings), 'length_m': round(sum(m['length_m'] for m in markings if m['type'] == t), 1)} for t in types}
    summary['crosswalk']['areas'] = len(crosswalks)
    summary['route_m'] = route_m
    summary['route_fraction_with_line_left'] = round(sides['left'] / max(route_m, 1), 3)
    summary['route_fraction_with_line_right'] = round(sides['right'] / max(route_m, 1), 3)
    meta = {'map': str(args.map.name), 'frame': 'enu', 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'source': str(source.relative_to(args.map)), 'method': 'intensity / local road median, oriented line filter, ridge tracing (see tools/extract_lane_markings.py)',
            'z': 'road surface height (1 m grid of road points), not the 2 m tree_voxels ground grid',
            'params': {'corridor_m': args.corridor_m, 'high': args.high, 'low': args.low, 'cell_m': CELL},
            'contrast_raster': {'file': 'road_intensity_contrast.png', 'origin_enu_m': [round(float(v), 3) for v in lo],
                                'cell_m': CELL, 'origin_is': 'south-west corner; image row 0 is the north edge', 'value': 'pixel / 100 = intensity / local road median'},
            'summary': summary, 'markings': markings,
            'crosswalks': [{'crosswalk_group': g, 'stripes': sum(m.get('crosswalk_group') == g for m in markings),
                            'outline_enu_m': [[round(float(x), 3), round(float(y), 3), round(float(z), 3)]
                                              for (x, y), z in zip(np.array(c), road_z(np.array(c)))]}
                           for g, c in enumerate(crosswalks)]}
    (out / 'lane_markings.json').write_text(json.dumps(meta, indent=1))
    cv2.imwrite(str(out / 'road_intensity_contrast.png'), np.clip(contrast * 100, 0, 255).astype(np.uint8)[::-1])
    zooms = [tuple(np.mean(s, 0)) for s in stop_xy] or [tuple(route[0, :2])]
    preview(out / 'lane_markings_preview.png', contrast, lo, route[:, :2], feats, crosswalks, stop_xy, zooms)
    print(f'{len(road)} road points near the route; {len(markings)} markings: ' +
          ', '.join(f"{t} {summary[t]['count']} ({summary[t]['length_m']} m)" for t in types))
    print(f"route {route_m} m: a line within 8 m on the left for {summary['route_fraction_with_line_left']:.0%}, "
          f"on the right for {summary['route_fraction_with_line_right']:.0%}")
    print(f'wrote {out}/lane_markings.json, lane_markings_preview.png, road_intensity_contrast.png')


if __name__ == '__main__':
    main()
