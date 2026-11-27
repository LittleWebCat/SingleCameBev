#!/usr/bin/env python3
"""Find trees in a saved map so RViz can hide them (display only; localization keeps the full map).

Trees are told apart by shape: leaves scatter LiDAR returns through a volume, while walls,
road, cars and poles return planar or linear surfaces. For every 0.5 m voxel more than 0.4 m
above the local ground, the covariance of the points in its 1.5 m neighbourhood gives
scattering = smallest / largest eigenvalue; a voxel is tree when most voxels within 1 m are
scattered and it is more than 1 m above ground (so low kerbs and lawns stay). Voxels near kept
stop signs / traffic lights are never tree, so a light inside a canopy stays visible.

Writes next to the map (MAP/gps_alignment/):
  pointcloud_map_enu_no_trees.pcd  the map without tree points (same x y z intensity format)
  tree_voxels.npz                  tree voxels, grown by one voxel for live scans (localization error),
                                   and the local ground height grid (map display colours height above it)
  trees_removed_preview.png        top-down before (trees green) / after
tools/localization_rviz_bridge.py uses them to publish the map and the live scan without trees.

  /usr/bin/python3 tools/remove_map_trees.py [--map maps/local_gps_recording_002]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from saved_map_source import read_pcd_xyz

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def ground_grid(xyz, cell_m=2.0):
    """Local ground: per-cell 10th-percentile height, holes filled, opened over ~30 m (drops buildings/trees).

    Returns (grid[row=y, col=x], origin xy, cell size); a point's cell is floor((xy - origin) / cell)."""
    origin = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - origin) / cell_m).astype(np.int64)
    w, h = ij.max(0) + 1
    cell = ij[:, 1] * w + ij[:, 0]
    order = np.lexsort((xyz[:, 2], cell))
    sorted_cells = cell[order]
    starts = np.r_[0, np.nonzero(np.diff(sorted_cells))[0] + 1]
    counts = np.diff(np.r_[starts, len(order)])
    low = np.full(w * h, np.nan)
    enough = counts >= 5
    low[sorted_cells[starts[enough]]] = xyz[order[starts[enough] + (counts[enough] * 0.1).astype(int)], 2]
    low = low.reshape(h, w)
    nearest = ndimage.distance_transform_edt(np.isnan(low), return_distances=False, return_indices=True)
    return ndimage.uniform_filter(ndimage.grey_opening(low[tuple(nearest)], size=(15, 15)), 5), origin, cell_m


def neighbour_offsets(dims, r):
    return [(dz * dims[1] + dy) * dims[0] + dx for dz in range(-r, r + 1) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def lookup(keys, query):
    j = np.searchsorted(keys, query)
    j[j >= len(keys)] = 0
    return j, keys[j] == query


def tree_voxels(points, above_ground, voxel_m):
    """Return (sorted voxel keys, origin, dims, tree flag per key) for above-ground points."""
    origin = points.min(0)
    ijk = np.floor((points - origin) / voxel_m).astype(np.int64)
    dims = ijk.max(0) + 3  # one empty voxel of margin on each side keeps neighbour keys unambiguous
    keys, inverse = np.unique(((ijk[:, 2] + 1) * dims[1] + ijk[:, 1] + 1) * dims[0] + ijk[:, 0] + 1, return_inverse=True)
    local = points - (ijk + 0.5) * voxel_m - origin
    sums = np.zeros((len(keys), 10))
    sums[:, 0] = np.bincount(inverse, minlength=len(keys))
    for a in range(3):
        sums[:, 1 + a] = np.bincount(inverse, weights=local[:, a], minlength=len(keys))
    for k, (a, b) in enumerate(PAIRS):
        sums[:, 4 + k] = np.bincount(inverse, weights=local[:, a] * local[:, b], minlength=len(keys))
    # 3x3x3 neighbourhood moments, each neighbour's coordinates shifted into this voxel's frame.
    hood = np.zeros_like(sums)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j, hit = lookup(keys, keys + (dz * dims[1] + dy) * dims[0] + dx)
                s, d = sums[j[hit]], np.array([dx, dy, dz]) * voxel_m
                n = s[:, 0]
                shifted = s.copy()
                shifted[:, 1:4] += n[:, None] * d
                for k, (a, b) in enumerate(PAIRS):
                    shifted[:, 4 + k] += d[a] * s[:, 1 + b] + d[b] * s[:, 1 + a] + n * d[a] * d[b]
                hood[hit] += shifted
    n = hood[:, 0]
    mean = hood[:, 1:4] / n[:, None]
    cov = np.empty((len(keys), 3, 3))
    for k, (a, b) in enumerate(PAIRS):
        cov[:, a, b] = cov[:, b, a] = hood[:, 4 + k] / n - mean[:, a] * mean[:, b]
    eig = np.maximum(np.linalg.eigvalsh(cov), 1e-9)  # ascending
    scattered = (eig[:, 0] / eig[:, 2] > 0.2) & (n >= 6)
    # Majority of scattered voxels within 1 m; voxel more than 1 m above ground.
    votes, seen = np.zeros(len(keys)), np.zeros(len(keys))
    for offset in neighbour_offsets(dims, 2):
        j, hit = lookup(keys, keys + offset)
        votes[hit] += scattered[j[hit]]
        seen[hit] += 1
    height = np.bincount(inverse, weights=above_ground, minlength=len(keys)) / sums[:, 0]
    return keys, inverse, origin, dims, (votes / seen > 0.5) & (height > 1.0)


def key_centres(keys, origin, dims, voxel_m):
    ijk = np.stack([keys % dims[0], (keys // dims[0]) % dims[1], keys // (dims[0] * dims[1])], 1) - 1
    return origin + (ijk + 0.5) * voxel_m


def protected(centres, landmarks, radius_m=2.0, above_m=2.0):
    """Columns around kept landmarks, from the ground up to just above the landmark."""
    keep = np.zeros(len(centres), bool)
    for x, y, z in landmarks:
        keep |= (np.hypot(centres[:, 0] - x, centres[:, 1] - y) < radius_m) & (centres[:, 2] < z + above_m)
    return keep


def kept_landmarks(map_dir):
    reviewed, found = map_dir / 'landmarks/landmarks_reviewed.json', map_dir / 'landmarks/landmarks.json'
    if reviewed.exists():
        return [l['position_enu_m'] for l in json.loads(reviewed.read_text())['landmarks'] if l['keep']], reviewed
    if found.exists():
        return [l['position_enu_m'] for l in json.loads(found.read_text())['landmarks'] if l['confirmed']], found
    return [], None


def write_pcd(path, points):
    header = (f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              f'WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(points, '<f4').tobytes())


def preview(path, xyz, tree, above, route):
    lo, cell = route.min(0) - 40, 0.4
    w, h = ((route.max(0) + 40 - lo) / cell).astype(int) + 1

    def render(layers):
        img = np.full((h, w, 3), 14, np.uint8)
        for mask, colour in layers:
            ij = ((xyz[mask, :2] - lo) / cell).astype(int)
            ok = (ij >= 0).all(1) & (ij[:, 0] < w) & (ij[:, 1] < h)
            img[h - 1 - ij[ok, 1], ij[ok, 0]] = colour
        r = ((route - lo) / cell).astype(int)
        cv2.polylines(img, [np.stack([r[:, 0], h - 1 - r[:, 1]], 1)[::5].reshape(-1, 1, 2)], False, (60, 220, 90), 2)
        return img

    ground, other = ~above, above & ~tree
    before = render([(ground, (70, 70, 70)), (other, (235, 235, 235)), (tree, (60, 160, 60))])
    after = render([(ground, (70, 70, 70)), (other, (235, 235, 235))])
    for img, text in ((before, 'before: trees green'), (after, 'after: trees removed')):
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), np.hstack([before, np.full((h, 8, 3), 255, np.uint8), after]))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=root / 'maps/local_gps_recording_002')
    parser.add_argument('--voxel-m', type=float, default=0.5)
    args = parser.parse_args()
    out = args.map / 'gps_alignment'
    source = out / 'pointcloud_map_enu.pcd'
    xyz = read_pcd_xyz(source)  # validates the x y z intensity float32 layout
    full = np.memmap(source, mode='r', dtype='<f4', offset=source.stat().st_size - len(xyz) * 16, shape=(len(xyz), 4))
    xyz = np.asarray(xyz, np.float64)
    route = np.genfromtxt(out / 'trajectory_enu.csv', delimiter=',', names=True)
    route = np.stack([route['x_m'], route['y_m']], 1)

    ground, ground_origin, ground_cell_m = ground_grid(xyz)
    ij = np.floor((xyz[:, :2] - ground_origin) / ground_cell_m).astype(np.int64)
    height = xyz[:, 2] - ground[ij[:, 1], ij[:, 0]]
    above = height > 0.4
    keys, inverse, origin, dims, tree = tree_voxels(xyz[above], height[above], args.voxel_m)
    landmarks, landmark_file = kept_landmarks(args.map)
    centres = key_centres(keys, origin, dims, args.voxel_m)
    tree &= ~protected(centres, landmarks)
    point_tree = np.zeros(len(xyz), bool)
    point_tree[np.nonzero(above)[0]] = tree[inverse]

    # Live scans: grow by one voxel to absorb localization/deskew error, then re-protect landmarks.
    tree_keys = keys[tree]
    grown = np.unique(np.concatenate([tree_keys + o for o in neighbour_offsets(dims, 1)]))
    grown = grown[~protected(key_centres(grown, origin, dims, args.voxel_m), landmarks)]
    np.savez_compressed(out / 'tree_voxels.npz', voxel_m=args.voxel_m, origin=origin, dims=dims,
                        live_keys=grown, map_keys=tree_keys, landmarks=np.array(landmarks, float).reshape(-1, 3),
                        ground=ground.astype(np.float32), ground_origin=ground_origin, ground_cell_m=ground_cell_m)
    write_pcd(out / 'pointcloud_map_enu_no_trees.pcd', full[~point_tree])
    preview(out / 'trees_removed_preview.png', xyz, point_tree, above, route)
    print(f'{point_tree.sum()} of {len(xyz)} map points are trees ({point_tree.mean() * 100:.1f}%); '
          f'{len(tree_keys)} tree voxels ({len(grown)} grown for live scans); '
          f'{len(landmarks)} landmark columns protected from {landmark_file.name if landmark_file else "none"}')
    print(f'wrote {out}/pointcloud_map_enu_no_trees.pcd, tree_voxels.npz, trees_removed_preview.png')


if __name__ == '__main__':
    main()
