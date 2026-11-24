#!/usr/bin/env python3
"""Import the multi-camera rig calibration (all_cameras_intrinsics/extrinsics JSON) per camera.

For every camera present in both files, writes into OUTPUT/:
  <camera>_<W>x<H>.yaml   ROS camera_info-style intrinsics at the calibrated resolution
                          (identity R, P = [K|0]; never resized)
  <camera>_T_os_sensor.yaml  extrinsics in the format camera_lidar_box_fusion loads:
                          T_camera_from_os_sensor from the file's T_cam_lidar (its LiDAR frame
                          is the Ouster sensor frame, x forward / y left / z up), plus the driver
                          os_sensor <- os_lidar transform and the composed T_camera_from_os_lidar.
Frames: camera_<camera>_optical_frame. Copies both source files and writes manifest.json.

  python3 tools/import_rig_calibration.py INTRINSICS.json EXTRINSICS.json config/calibration/rig_20261124
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

# Driver transform os_sensor <- os_lidar (fixed Ouster mounting used throughout this project).
SENSOR_FROM_LIDAR = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.038195], [0, 0, 0, 1]], float)


def matrix_yaml(m):
    return [[float(v) for v in row] for row in np.asarray(m)]


def quat_xyzw(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return [float(x), float(y), float(z), float(w)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('intrinsics', type=Path)
    p.add_argument('extrinsics', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    intr = json.loads(a.intrinsics.read_text())['cameras']
    extr_doc = json.loads(a.extrinsics.read_text())
    extr = extr_doc['cameras']
    a.output.mkdir(parents=True, exist_ok=False)
    for src in (a.intrinsics, a.extrinsics):
        shutil.copy2(src, a.output / src.name)
    manifest = {'source_files': {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in (a.intrinsics, a.extrinsics)},
                'lidar_frame_interpretation': 'os_sensor (x forward, y left, z up); extrinsics file: ' + extr_doc.get('lidar_frame', ''),
                'cameras': {}}
    for name, cam in intr.items():
        if name not in extr or not extr[name].get('valid', False):
            print(f'skip {name}: no valid extrinsics')
            continue
        e = extr[name]
        w, h = int(cam['image_width']), int(cam['image_height'])
        if (w, h) != (int(e['image_width']), int(e['image_height'])):
            raise SystemExit(f'{name}: intrinsics {w}x{h} and extrinsics resolution differ')
        k = np.array(cam['camera_matrix'], float)
        if not np.allclose(k, np.array(e['intrinsics']['camera_matrix'], float)):
            raise SystemExit(f'{name}: extrinsics were fitted with different intrinsics')
        frame = f'camera_{name}_optical_frame'
        intrinsics_file = a.output / f'{name}_{w}x{h}.yaml'
        intrinsics_file.write_text(yaml.safe_dump({
            'image_width': w, 'image_height': h, 'camera_name': name, 'distortion_model': 'plumb_bob',
            'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in k.ravel()]},
            'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': [float(v) for v in cam['dist_coeffs']]},
            'rectification_matrix': {'rows': 3, 'cols': 3, 'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
            'projection_matrix': {'rows': 3, 'cols': 4, 'data': [float(k[0, 0]), 0.0, float(k[0, 2]), 0.0, 0.0,
                                                                  float(k[1, 1]), float(k[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]},
        }, sort_keys=False))
        cam_from_sensor = np.array(e['T_cam_lidar'], float)
        r = cam_from_sensor[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
            raise SystemExit(f'{name}: T_cam_lidar rotation is not orthonormal')
        cam_from_lidar = cam_from_sensor @ SENSOR_FROM_LIDAR
        extrinsics_file = a.output / f'{name}_T_os_sensor.yaml'
        extrinsics_file.write_text(yaml.safe_dump({
            'schema_version': 1, 'calibrated': False, 'calibration_status': 'external_candidate_unvalidated',
            'units': 'm', 'direction': 'lidar_to_camera', 'selected_hypothesis': 'B',
            'selection_basis': 'Rig calibration file: LiDAR frame is the Ouster sensor frame (x forward, y left, z up)',
            'calibration_date': e.get('calibrated'), 'device_by_path': e.get('device_by_path'),
            'fit_quality': {'boards_used': e.get('boards_used'), 'plane_rms_mm_median': float(np.median(e.get('plane_rms_mm', [np.nan]))),
                            'pattern_score_min_max': e.get('pattern_score_min_max'),
                            'intrinsics_rms_reprojection_px': cam.get('rms_reprojection_error_px')},
            'observed_driver_transform': {'source_frame': 'os_lidar', 'target_frame': 'os_sensor', 'matrix': matrix_yaml(SENSOR_FROM_LIDAR)},
            'transforms': {
                'T_camera_from_os_sensor': {'calibrated': False, 'source_frame': 'os_sensor', 'target_frame': frame,
                                            'translation_m': [float(v) for v in cam_from_sensor[:3, 3]],
                                            'quaternion_xyzw': quat_xyzw(r), 'matrix': matrix_yaml(cam_from_sensor),
                                            'inverse_matrix': matrix_yaml(np.linalg.inv(cam_from_sensor))},
                'T_camera_from_os_lidar': {'calibrated': False, 'source_frame': 'os_lidar', 'target_frame': frame,
                                           'matrix': matrix_yaml(cam_from_lidar)}},
        }, sort_keys=False))
        pos = np.linalg.inv(cam_from_sensor)[:3, 3]
        axis = np.linalg.inv(cam_from_sensor)[:3, 2]
        manifest['cameras'][name] = {'device_by_path': cam['device_by_path'], 'resolution': [w, h], 'frame': frame,
                                     'intrinsics': intrinsics_file.name, 'extrinsics': extrinsics_file.name,
                                     'position_in_os_sensor_m': [round(float(v), 4) for v in pos],
                                     'optical_axis_yaw_deg': round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1),
                                     'hfov_deg': cam.get('fov_horizontal_deg')}
        print(f'{name:13s} {w}x{h}  at x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f} m, looking {manifest["cameras"][name]["optical_axis_yaw_deg"]:+6.1f} deg '
              f'(HFOV {cam.get("fov_horizontal_deg")})  calibrated {e.get("calibrated")}')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
