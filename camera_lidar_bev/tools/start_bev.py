#!/usr/bin/python3
"""Start missing project streams, reuse existing nodes, and supervise owned children."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def matching_processes(executable, domain):
    matches = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            arguments = (proc / 'cmdline').read_bytes().split(b'\0')
            if not arguments:
                continue
            direct = Path(os.fsdecode(arguments[0])).name == executable
            script = (len(arguments) > 1 and executable.endswith('.py') and
                      Path(os.fsdecode(arguments[1])).name == executable)
            if not direct and not script:
                continue
            values = (proc / 'environ').read_bytes().split(b'\0')
            actual = next((os.fsdecode(v).split('=', 1)[1] for v in values
                           if v.startswith(b'ROS_DOMAIN_ID=')), '0')
            if actual != domain:
                raise RuntimeError(f'{executable} already runs in ROS domain {actual}; expected {domain}.')
            matches.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) > 1:
        raise RuntimeError(f'Multiple {executable} processes found: {matches}; resolve duplicates first.')
    return matches


def preflight(root, camera_labels=False, lidar_3d_boxes=False, camera_lidar_boxes=False,
              combined_view=False, road_lanes=True):
    if combined_view:
        required = ['bev_sensor_bridge/lib/bev_sensor_bridge/camera_bridge',
                    'bev_lidar_detection/lib/bev_lidar_detection/lidar_vehicle_detector',
                    'bev_visualization/lib/bev_visualization/camera_object_detector',
                    'bev_visualization/lib/bev_visualization/camera_lidar_box_fusion']
        if road_lanes and (not (root.parent / '.venv/bin/python').is_file() or not (
                root / 'models/road_lanes_001/twinlitenet_fp16.engine').is_file()):
            raise RuntimeError('Road/lane model is missing; see docs/road_lane_detection.md.')
    elif camera_lidar_boxes:
        required = ['bev_lidar_detection/lib/bev_lidar_detection/lidar_vehicle_detector',
                    'bev_sensor_bridge/lib/bev_sensor_bridge/camera_bridge',
                    'bev_visualization/lib/bev_visualization/camera_object_detector',
                    'bev_visualization/lib/bev_visualization/camera_lidar_box_fusion']
    elif lidar_3d_boxes:
        required = ['bev_lidar_detection/lib/bev_lidar_detection/lidar_vehicle_detector']
    else:
        required = ['bev_sensor_bridge/lib/bev_sensor_bridge/camera_bridge',
                    'bev_visualization/lib/bev_visualization/camera_rgb_cloud',
                    'bev_geometry/lib/bev_geometry/ground_bev_node']
    if camera_labels and not camera_lidar_boxes:
        required += ['bev_visualization/lib/bev_visualization/camera_object_detector',
                     'bev_visualization/lib/bev_visualization/camera_object_bev']
    for relative in required:
        if not (root / 'install' / relative).is_file():
            raise RuntimeError('Build the project first: colcon build --packages-up-to bev_bringup '
                               '--symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 '
                               '-DCMAKE_BUILD_TYPE=Release')
    if camera_labels or camera_lidar_boxes or combined_view:
        model = root / 'models/camera_objects_001'
        engine = model / 'yolo11n_fp16_raw.engine'
        try:
            provenance = json.loads((model / 'provenance.json').read_text())
            expected = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle',
                        '5': 'bus', '7': 'truck'}
            if any(provenance['classes'].get(key) != value for key, value in expected.items()):
                raise ValueError('model class names differ from the configured COCO road users')
            if hashlib.sha256(engine.read_bytes()).hexdigest() != provenance['raw_engine_sha256']:
                raise ValueError('engine hash differs from its recorded export')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f'Camera model is missing or differs from its provenance: {error}. '
                               'See docs/camera_objects.md.') from error
    if lidar_3d_boxes or camera_lidar_boxes or combined_view:
        model = root / 'models/lidar_objects_001'
        engine = model / 'pointpillar_fp16.plan'
        try:
            provenance = json.loads((model / 'provenance.json').read_text())
            if provenance['engine_file'] != engine.name:
                raise ValueError('recorded engine filename differs from the configured engine')
            if provenance['classes'] != ['Car', 'Pedestrian', 'Cyclist']:
                raise ValueError('model classes differ from the expected KITTI class mapping')
            if hashlib.sha256(engine.read_bytes()).hexdigest() != provenance['engine_sha256']:
                raise ValueError('engine hash differs from its recorded export')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f'LiDAR model is missing or differs from its provenance: {error}. '
                               'See docs/lidar_vehicle_detection.md.') from error
    try:
        limit = int(Path('/proc/sys/net/core/rmem_max').read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError('Cannot verify host UDP receive limit; run this launcher on the sensor host.') from error
    if limit < 16777216:
        raise RuntimeError(
            'DDS needs a 16 MiB receive buffer. Install persistent settings once:\n'
            f'sudo bash {root / "tools/setup_dds_transport.sh"}\n'
            'Or apply temporary settings for this boot:\n'
            'sudo sysctl -w net.core.rmem_max=16777216 net.core.rmem_default=4194304 '
            'net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728')


def stop_owned(children, grace_s=15.):
    for _, process in reversed(children):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_s
    for _, process in reversed(children):
        try:
            process.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-rviz', action='store_true')
    parser.add_argument('--no-occupancy-view', action='store_true')
    parser.add_argument('--camera-labels', action='store_true',
                        help='also run camera object labels and LiDAR-supported observed boxes')
    parser.add_argument('--bev-only', action='store_true',
                        help='show ground and object boxes in one BEV window; enables camera labels')
    parser.add_argument('--object-points-only', action='store_true',
                        help='show only labeled object points in one BEV window; enables camera labels')
    parser.add_argument('--vehicle-bev-boxes', action='store_true',
                        help='show labeled points and flat vehicle boxes in BEV; enables camera labels')
    parser.add_argument('--lidar-3d-boxes', action='store_true',
                        help='run trained LiDAR car/person detection with flat BEV footprints and size labels')
    parser.add_argument('--camera-lidar-boxes', action='store_true',
                        help='show LiDAR car/person points with only camera-matched boxes in flat BEV')
    parser.add_argument('--combined-view', action='store_true',
                        help='show image, LiDAR projection, YOLO 2D, LiDAR 3D boxes and road/lane overlays beside an 80 m BEV')
    parser.add_argument('--localize-map', type=Path,
                        help='with --combined-view: use independent live ENU odometry on this saved map')
    parser.add_argument('--gps-alignment', type=Path,
                        help='GPS alignment directory for --localize-map; default MAP/gps_alignment')
    parser.add_argument('--no-road-lanes', action='store_true',
                        help='with --combined-view: omit road/lane inference')
    parser.add_argument('--lidar-mode', choices=('1024x20', '2048x10'),
                        help='mode for a missing driver; existing drivers are reused without reconfiguration')
    parser.add_argument('--range-m', type=float, default=80.)
    parser.add_argument('--max-points', type=int, default=30000)
    parser.add_argument('--max-map-points', type=int, default=100000)
    args = parser.parse_args(argv)
    if (args.localize_map or args.gps_alignment or args.no_road_lanes) and not args.combined_view:
        parser.error('--localize-map, --gps-alignment and --no-road-lanes require --combined-view')
    if args.gps_alignment and not args.localize_map:
        parser.error('--gps-alignment requires --localize-map')
    if not 0 < args.range_m <= 200 or args.max_points < 1 or args.max_map_points < 1:
        parser.error('--range-m must be within (0, 200]; point limits must be positive')
    if args.combined_view and (args.camera_lidar_boxes or args.lidar_3d_boxes or
                              args.camera_labels or args.bev_only or args.object_points_only or args.vehicle_bev_boxes):
        parser.error('--combined-view selects the combined pipeline; use it without other display flags')
    if args.combined_view:
        args.no_rviz = True
        args.no_occupancy_view = True
    if args.camera_lidar_boxes and (args.lidar_3d_boxes or args.camera_labels or args.bev_only or
                                   args.object_points_only or args.vehicle_bev_boxes):
        parser.error('--camera-lidar-boxes selects the combined preview; use it without other display flags')
    if args.lidar_3d_boxes and (args.camera_labels or args.bev_only or
                              args.object_points_only or args.vehicle_bev_boxes):
        parser.error('--lidar-3d-boxes selects the LiDAR-only preview; use it without camera preview flags')
    if args.lidar_3d_boxes or args.camera_lidar_boxes:
        args.no_occupancy_view = True
    if args.bev_only or args.object_points_only or args.vehicle_bev_boxes:
        args.camera_labels = True
        args.no_occupancy_view = True
    root = Path(__file__).resolve().parents[1]
    domain = os.environ.get('ROS_DOMAIN_ID', '1')
    if not domain.isdecimal() or not 0 <= int(domain) <= 231:
        parser.error('ROS_DOMAIN_ID must be an integer from 0 to 231')
    try:
        preflight(root, args.camera_labels, args.lidar_3d_boxes, args.camera_lidar_boxes,
                  args.combined_view, not args.no_road_lanes)
        camera = root / 'config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml'
        commands = ([
            ('os_driver', ['/usr/bin/python3', str(root / 'tools/start_lidar.py'), '--startup-timeout', '45']),
            ('lidar_vehicle_detector', ['ros2', 'launch', 'bev_bringup',
                                        'lidar_vehicle_detector.launch.py', 'rviz:=false']+
                                       (['max_forward_m:=69.12'] if args.combined_view else [])),
        ] if args.lidar_3d_boxes or args.camera_lidar_boxes or args.combined_view else [
            ('camera_bridge', ['ros2', 'launch', 'bev_bringup', 'front_camera.launch.py',
                               'width:=1024', 'height:=768', 'fps:=30.0', f'calibration_file:={camera}']),
            ('os_driver', ['/usr/bin/python3', str(root / 'tools/start_lidar.py'), '--startup-timeout', '45']),
            ('camera_rgb_cloud', ['ros2', 'launch', 'bev_bringup', 'camera_bev.launch.py', 'rviz:=false']),
            ('ground_bev_node', ['ros2', 'launch', 'bev_bringup', 'ground_bev.launch.py', 'rviz:=false']),
        ])
        if args.camera_lidar_boxes or args.combined_view:
            commands += [
                ('camera_bridge', ['ros2', 'launch', 'bev_bringup', 'front_camera.launch.py',
                                   'width:=1024', 'height:=768', 'fps:=30.0', f'calibration_file:={camera}']),
                ('camera_object_detector', ['ros2', 'launch', 'bev_bringup',
                  'camera_objects.launch.py', 'detector:=true', 'fusion:=false', 'rviz:=false']),
                ('camera_lidar_box_fusion', ['ros2', 'launch', 'bev_bringup',
                                           'camera_lidar_boxes.launch.py', 'rviz:=false']),
            ]
        if args.combined_view:
            if not args.no_road_lanes:
                commands += [('road_lane_detector.py', [str(root.parent / '.venv/bin/python'),
                                          str(root / 'tools/road_lane_detector.py'),
                                          '--engine', str(root / 'models/road_lanes_001/twinlitenet_fp16.engine')])]
            commands += [
                ('combined_bev_live.py', ['/usr/bin/python3', str(root / 'tools/combined_bev_live.py'),
                                         '--intrinsics', str(camera), '--extrinsics',
                                         str(root / 'config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml'),
                                         '--range-m', str(args.range_m), '--max-points', str(args.max_points),
                                         '--max-map-points', str(args.max_map_points)] +
                                         (['--localize-map', str(args.localize_map.resolve()),
                                           '--gps-alignment', str((args.gps_alignment or
                                                                   args.localize_map/'gps_alignment').resolve())]
                                          if args.localize_map else [])),
            ]
        if args.camera_labels:
            commands += [
                ('camera_object_detector', ['ros2', 'launch', 'bev_bringup',
                  'camera_objects.launch.py', 'detector:=true', 'fusion:=false', 'rviz:=false']),
                ('camera_object_bev', ['ros2', 'launch', 'bev_bringup',
                  'camera_objects.launch.py', 'detector:=false', 'fusion:=true', 'rviz:=false']),
            ]
        actions = []
        for executable, command in commands:
            if executable == 'os_driver' and args.lidar_mode:
                command += ['--params-file', str(root / f'config/sensors/ouster_{args.lidar_mode}.yaml')]
            existing = matching_processes(executable, domain)
            if executable == 'os_driver':
                existing = existing or matching_processes('start_lidar.py', domain)
            if existing:
                if executable == 'combined_bev_live.py' and args.localize_map:
                    raise RuntimeError('The combined viewer already runs; close it before opening live localization.')
                print(f'Reuse {executable} PID {existing[0]} in ROS domain {domain}.', flush=True)
            else:
                actions.append((executable, command))
        if not args.no_rviz:
            existing = matching_processes('rviz2', domain)
            if existing:
                raise RuntimeError('RViz is already open. Close it to open the selected preview, '
                                   'or use --no-rviz to start only missing processors.')
            view = ('camera_lidar_boxes.rviz' if args.camera_lidar_boxes else
                    'lidar_vehicle_boxes.rviz' if args.lidar_3d_boxes else
                    'vehicle_bev_boxes.rviz' if args.vehicle_bev_boxes else
                    'object_points_only.rviz' if args.object_points_only else
                    'bev_only.rviz' if args.bev_only else
                    'camera_objects.rviz' if args.camera_labels else 'ground_bev.rviz')
            actions.append(('rviz2', ['rviz2', '-d', str(root / 'config/rviz' / view),
                                      '--ros-args', '-r', '__node:=ground_bev_view']))
        if not args.no_occupancy_view:
            existing = matching_processes('view_occupancy.py', domain)
            if existing:
                print(f'Reuse occupancy view PID {existing[0]}.', flush=True)
            else:
                actions.append(('view_occupancy.py', ['/usr/bin/python3', str(root / 'tools/view_occupancy.py')]))
        for name, command in actions:
            print(f'{"Would start" if args.dry_run else "Start"} {name}: {command}', flush=True)
        if args.dry_run:
            return 0
    except RuntimeError as error:
        print(f'BEV startup stopped: {error}', flush=True)
        return 2

    children = []
    try:
        for name, command in actions:
            children.append((name, subprocess.Popen(command, cwd=root, start_new_session=True)))
        if args.combined_view:
            print('Combined view: camera with LiDAR projection and YOLO 2D boxes beside '
                  f'{args.range_m:g} m flat LiDAR BEV footprints. '
                  'Keys 1-8 toggle perception layers; 9 toggles the map.', flush=True)
        elif args.camera_lidar_boxes:
            print('Camera + LiDAR BEV: native car/person points and only camera-matched boxes, '
                  'with learned dimensions/yaw and separate model scores. The view stays flat; '
                  'calibration is provisional.', flush=True)
        elif args.lidar_3d_boxes:
            print('LiDAR car/person detection: orange cars, magenta people, flat footprints and predicted '
                  'length/width/height labels. KITTI-to-Ouster accuracy is provisional.', flush=True)
        elif args.vehicle_bev_boxes:
            print('Vehicle BEV: orange vehicle points and flat footprint boxes, magenta person points. '
                  'Boxes enclose observed LiDAR returns.', flush=True)
        elif args.object_points_only:
            print('Object points: person magenta, vehicles orange.', flush=True)
        else:
            print('Ground preview: ground green, obstacles red, unknown gray.', flush=True)
            if args.camera_labels:
                print('Camera labels: person magenta, vehicles orange. '
                      'Boxes show observed LiDAR extents.', flush=True)
        print(('Live saved-map localization is enabled. ' if args.localize_map else
               'Keep the rig stationary. ') + 'Ctrl+C stops only started children.', flush=True)
        closed_viewers = set()
        while children:
            for name, process in children:
                if process.poll() is None:
                    continue
                if name in ('rviz2', 'view_occupancy.py', 'combined_bev_live.py'):
                    if name == 'combined_bev_live.py' and args.localize_map:
                        if process.returncode != 0:
                            raise RuntimeError(f'{name} exited with {process.returncode}; inspect its output.')
                        print('Localized viewer closed; stopping owned processors.', flush=True)
                        return 0
                    if name not in closed_viewers:
                        print(f'{name} closed; processors continue. Reopen the viewer or press Ctrl+C.', flush=True)
                        closed_viewers.add(name)
                else:
                    raise RuntimeError(f'{name} launcher exited with {process.returncode}; inspect its output.')
            time.sleep(.25)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as error:
        print(f'BEV startup stopped: {error}', flush=True)
        return 2
    finally:
        stop_owned(children)


if __name__ == '__main__':
    raise SystemExit(main())
