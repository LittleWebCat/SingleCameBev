"""ROS-free replay dataset for object prediction: one .npz per recorded drive (numpy only).

Written on the car's computer by tools/export_prediction_dataset.py, read anywhere by
tools/eval_prediction.py. Every LiDAR scan with a localization pose is one frame; its CenterPoint
detections of trackable classes are already in the map frame (enu, metres), so the car's own motion is
removed. Arrays:
  frame_t      (F,)   scan time, s                     ego_xyz_yaw (F, 4) car (LiDAR) x, y, z and heading
  det_frame    (D,)   frame index of each detection    det_class   (D,)   index into class_names
  det_xyz      (D, 3) box centre, enu                  det_size    (D, 3) box length, width, height
  det_yaw      (D,)   box heading, enu (front/back ambiguous)   det_score (D,) detector confidence
  route_xyz    (N, 3) mapped route (LiDAR positions of the mapping drive), enu
  class_names, meta (JSON: source bag, map, export settings)
"""
import json

import numpy as np

GROUP_OF = {'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle',
            'construction_vehicle': 'vehicle', 'bicycle': 'cyclist', 'motorcycle': 'cyclist', 'person': 'pedestrian'}


def save_dataset(path, frame_t, ego, detections, route_xyz, meta):
    """detections: [(frame index, class name, xyz, size, yaw, score)]."""
    names = sorted(GROUP_OF)
    d = list(zip(*detections)) if detections else [[]] * 6
    np.savez_compressed(
        path, frame_t=np.asarray(frame_t, np.float64), ego_xyz_yaw=np.asarray(ego, np.float32).reshape(-1, 4),
        det_frame=np.asarray(d[0], np.int32), det_class=np.asarray([names.index(c) for c in d[1]], np.int8),
        det_xyz=np.asarray(d[2], np.float32).reshape(-1, 3), det_size=np.asarray(d[3], np.float32).reshape(-1, 3),
        det_yaw=np.asarray(d[4], np.float32), det_score=np.asarray(d[5], np.float32),
        route_xyz=np.asarray(route_xyz, np.float32), class_names=np.array(names), meta=np.array(json.dumps(meta)))


def load_dataset(path, min_score=0.35):
    """(frames, route_xyz, meta, ego): frames = [(t, [(group, class, xyz, size, yaw)])] as the live node sees them."""
    z = np.load(path)
    names = [str(n) for n in z['class_names']]
    keep = z['det_score'] >= min_score
    order = np.argsort(z['det_frame'][keep], kind='stable')
    f_idx = z['det_frame'][keep][order]
    cls, xyz, size, yaw = (z[k][keep][order] for k in ('det_class', 'det_xyz', 'det_size', 'det_yaw'))
    bounds = np.searchsorted(f_idx, np.arange(len(z['frame_t']) + 1))
    frames = []
    for i, t in enumerate(z['frame_t']):
        dets = []
        for j in range(bounds[i], bounds[i + 1]):
            name = names[cls[j]]
            dets.append((GROUP_OF[name], name, xyz[j].astype(np.float64), tuple(float(v) for v in size[j]), float(yaw[j])))
        frames.append((float(t), dets))
    return frames, z['route_xyz'].astype(np.float64), json.loads(str(z['meta'])), z['ego_xyz_yaw']
