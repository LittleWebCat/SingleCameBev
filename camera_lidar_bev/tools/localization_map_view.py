#!/usr/bin/python3
"""Compare live GPS and LiDAR localization on a north-up saved ENU map.

GPS marks the receiver/antenna reference, whose offset from os_sensor has not
been measured. Its ellipse is receiver-reported uncertainty, not measured error.
No recorded trajectory is used to generate a position. ROS imports are lazy so
geometry, display and timing can be exercised without a running ROS graph.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import cv2
import numpy as np

from align_local_map_gps import geographic_to_enu, sha256_file
from saved_map_source import quaternion_rotation, read_pcd_xyz

GPS_TOPIC = '/sensing/gnss/fix'
POSE_TOPIC = '/camera_lidar_bev/localization/odometry'
LOCALIZER_STATUS_TOPIC = '/camera_lidar_bev/localization/status'
VIEW_STATUS_TOPIC = '/camera_lidar_bev/localization/view_status'
VIEW_IMAGE_TOPIC = '/camera_lidar_bev/localization/map_image'


def load_map_alignment(directory):
    """Verify the saved anchor before converting either stream to ENU."""
    directory = Path(directory).expanduser().resolve()
    if (directory/'.incomplete').exists():
        raise ValueError('GPS alignment export is incomplete')
    data = json.loads((directory/'alignment.json').read_text())
    source = data.get('source', {})
    map_dir = Path(source['map_dir']).expanduser().resolve()
    manifest = json.loads((map_dir/'run_manifest.json').read_text())
    mapping = json.loads((map_dir/'mapping_report.json').read_text())
    if (Path(manifest['bag']).resolve() != Path(source['bag']).resolve() or
            mapping.get('failed_input') is not False or mapping.get('map_frame') != 'map' or
            mapping.get('pose_child_frame') != 'os_sensor'):
        raise ValueError('alignment and source-map provenance do not agree')
    for name, path in (('pointcloud', map_dir/'pointcloud_map.pcd'),
                       ('trajectory', map_dir/'trajectory.csv'),
                       ('run_manifest', map_dir/'run_manifest.json'),
                       ('bag_metadata', Path(source['bag'])/'metadata.yaml')):
        if source.get(name+'_sha256') and sha256_file(path) != source[name+'_sha256']:
            raise ValueError(f'alignment source {name} checksum changed')
    matrix = np.asarray(data['transform']['T_enu_from_map'], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('map alignment must be a rigid transform without scale')
    origin = data['origin_wgs84']
    origin = np.array([origin['latitude_deg'], origin['longitude_deg'], origin['altitude_m']], float)
    geographic_to_enu(origin[None], origin)  # Reject invalid/nonfinite coordinates.
    return data, matrix, origin, map_dir/'pointcloud_map.pcd'


def enu_rotation(latitude_deg, longitude_deg):
    """ECEF -> local east/north/up rotation at the given WGS84 position."""
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    sp, cp, sl, cl = math.sin(latitude), math.cos(latitude), math.sin(longitude), math.cos(longitude)
    return np.array([[-sl, cl, 0.], [-sp*cl, -sp*sl, cp], [cp*cl, cp*sl, sp]])


def gps_observation(message, origin):
    """Return original-stamp GPS position/covariance; never invent heading."""
    stamp = int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)
    geographic = np.array([message.latitude, message.longitude, message.altitude], float)
    if message.status.status < 0 or stamp <= 0 or not np.isfinite(geographic).all():
        return None, 'no usable GPS fix'
    try:
        position = geographic_to_enu(geographic[None], origin)[0]
    except ValueError:
        return None, 'invalid GPS geographic coordinates'
    covariance = np.asarray(message.position_covariance, float).reshape(3, 3)
    known = (int(message.position_covariance_type) > 0 and np.isfinite(covariance).all() and
             np.allclose(covariance, covariance.T, atol=1e-6) and
             np.linalg.eigvalsh(covariance).min() >= -1e-6)
    if known:
        # NavSatFix covariance is ENU at the fix, not necessarily at map origin.
        rotation = enu_rotation(*origin[:2])@enu_rotation(*geographic[:2]).T
        covariance = rotation@covariance@rotation.T
    return {'stamp_ns': int(stamp), 'position': position, 'covariance': covariance if known else None,
            'fix_status': int(message.status.status), 'heading': None}, None


def lidar_observation(message, matrix, accepted_frames=('enu', 'local_enu')):
    stamp = int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)
    if stamp <= 0 or message.child_frame_id != 'os_sensor':
        return None, 'invalid LiDAR pose stamp or child frame'
    pose = message.pose.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z], float)
    try:
        rotation = quaternion_rotation([pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w])
    except ValueError:
        return None, 'invalid LiDAR pose orientation'
    if not np.isfinite(position).all():
        return None, 'nonfinite LiDAR position'
    if message.header.frame_id == 'map':
        position = matrix[:3, :3]@position+matrix[:3, 3]
        rotation = matrix[:3, :3]@rotation
    elif message.header.frame_id not in accepted_frames:
        return None, f'unsupported LiDAR frame: {message.header.frame_id}'
    return {'stamp_ns': int(stamp), 'position': position,
            'heading': math.atan2(rotation[1, 0], rotation[0, 0])}, None


def freshness(observation, now_ns, wall_now, max_source_age_s=1., max_receive_age_s=2.):
    """Use ROS source time and monotonic receipt independently, including pauses."""
    if observation is None:
        return False, 'waiting for data', None
    age = (now_ns-observation['stamp_ns'])*1e-9
    if now_ns <= 0:
        return False, 'waiting for ROS clock', age
    if age < -.05:
        return False, 'source timestamp is in the future', age
    if age > max_source_age_s:
        return False, 'source data stale', age
    if wall_now-observation['received_wall'] > max_receive_age_s:
        return False, 'stream paused or disconnected', age
    return True, 'fresh', age


def ellipse_geometry(covariance_xy, confidence_scale=math.sqrt(5.991464547)):
    """95% 2-D Gaussian contour from a covariance in square metres."""
    covariance = np.asarray(covariance_xy, float)
    if (covariance.shape != (2, 2) or not np.isfinite(covariance).all() or
            not np.allclose(covariance, covariance.T, atol=1e-6)):
        raise ValueError('ellipse covariance must be finite, symmetric 2 x 2')
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-6:
        raise ValueError('ellipse covariance must be positive semidefinite')
    order = np.argsort(values)[::-1]
    axes = confidence_scale*np.sqrt(np.maximum(values[order], 0.))
    major = vectors[:, order[0]]
    return axes, math.atan2(major[1], major[0])


class LocalizationMapRenderer:
    """Rasterize a bounded saved-map sample once; draw only dots/paths per frame."""
    def __init__(self, xyz_enu, width=1400, height=900):
        self.width, self.height, self.side_width = int(width), int(height), 340
        if self.width < 900 or self.height < 500:
            raise ValueError('map display must be at least 900 x 500 pixels')
        xyz = np.asarray(xyz_enu, float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if not len(xyz):
            raise ValueError('saved map has no finite points')
        self.plot_left, self.plot_top = 55, 75
        self.plot_width = self.width-self.side_width-self.plot_left-25
        self.plot_height = self.height-self.plot_top-65
        minimum, maximum = xyz[:, :2].min(axis=0)-20., xyz[:, :2].max(axis=0)+20.
        extent = np.maximum(maximum-minimum, 1.)
        self.scale = min(self.plot_width/extent[0], self.plot_height/extent[1])
        self.center = (minimum+maximum)/2
        self.pixel_center = np.array([self.plot_left+self.plot_width/2,
                                      self.plot_top+self.plot_height/2])
        self.base = np.full((self.height, self.width, 3), (24, 20, 17), np.uint8)
        self.base[:, self.width-self.side_width:] = (34, 29, 24)
        spacing = 20. if extent.max() < 250 else 50. if extent.max() < 650 else 100.
        for axis in (0, 1):
            for value in np.arange(math.floor(minimum[axis]/spacing)*spacing, maximum[axis], spacing):
                ends = np.array([minimum, maximum])
                ends[:, axis] = value
                pixels = self.pixels(ends)
                cv2.line(self.base, tuple(pixels[0]), tuple(pixels[1]), (39, 35, 30), 1)
                point = pixels[0] if axis == 0 else pixels[1]
                cv2.putText(self.base, f'{value:.0f}', tuple(point+(np.array([-15, 18]) if axis == 0 else [-32, 4])),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (150, 150, 150), 1, cv2.LINE_AA)
        pixels = self.pixels(xyz[:, :2])
        valid = ((pixels[:, 0] >= self.plot_left) & (pixels[:, 0] < self.width-self.side_width-10) &
                 (pixels[:, 1] >= self.plot_top) & (pixels[:, 1] < self.height-40))
        pixels = pixels[valid]
        self.base[pixels[:, 1], pixels[:, 0]] = (96, 96, 92)
        cv2.putText(self.base, 'Saved map | NORTH UP', (25, 36), cv2.FONT_HERSHEY_SIMPLEX, .85,
                    (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(self.base, 'East (m)', (self.plot_left+self.plot_width//2-25, self.height-17),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(self.base, 'North', (8, 61), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1)
        self.gps_color, self.lidar_color = (40, 170, 255), (255, 220, 45)

    def pixels(self, xy):
        xy = np.asarray(xy)
        return np.rint((xy-self.center)*[self.scale, -self.scale]+self.pixel_center).astype(np.int32)

    def render(self, gps=None, lidar=None, gps_path=(), lidar_path=(), status=None):
        image = self.base.copy()
        status = status or {}
        mode = str(status.get('source_mode', 'unknown')).upper()
        source_label = {'REPLAY': 'REPLAY | recorded sensors', 'LIVE': 'LIVE | current sensors'}.get(mode, '')
        if source_label:
            cv2.putText(image, source_label, (self.width-self.side_width-350, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 210, 250), 2, cv2.LINE_AA)
        for path, color in ((gps_path, self.gps_color), (lidar_path, self.lidar_color)):
            if len(path) > 1:
                cv2.polylines(image, [self.pixels(np.asarray(path)[:, :2])], False, color, 1, cv2.LINE_AA)
        if gps is not None:
            point = tuple(self.pixels(gps['position'][:2]))
            covariance = gps.get('covariance')
            if covariance is not None:
                axes, angle = ellipse_geometry(covariance[:2, :2])
                axes_px = tuple(np.clip(np.rint(axes*self.scale), 1, 20000).astype(int))
                cv2.ellipse(image, point, axes_px, -math.degrees(angle), 0, 360, self.gps_color, 1, cv2.LINE_AA)
            cv2.circle(image, point, 7, self.gps_color, -1, cv2.LINE_AA)
        if lidar is not None:
            point = self.pixels(lidar['position'][:2])
            heading = lidar['heading']
            end = point+np.rint([math.cos(heading)*20, -math.sin(heading)*20]).astype(int)
            cv2.arrowedLine(image, tuple(point), tuple(end), self.lidar_color, 3, cv2.LINE_AA, tipLength=.4)
            cv2.circle(image, tuple(point), 5, self.lidar_color, -1, cv2.LINE_AA)
        left, y = self.width-self.side_width+18, 34
        lines = [
            ('GPS antenna/reference', self.gps_color),
            (status.get('gps_state', 'waiting for GPS'), (230, 230, 230)),
            (status.get('gps_age_text', ''), (175, 175, 175)),
            (status.get('gps_sigma_text', ''), (175, 175, 175)),
            ('', (0, 0, 0)),
            ('LiDAR os_sensor pose', self.lidar_color),
            (status.get('lidar_state', 'waiting for localizer'), (230, 230, 230)),
            (status.get('lidar_age_text', ''), (175, 175, 175)),
            (status.get('localizer_text', ''), (175, 175, 175)),
            (status.get('matching_text', ''), (175, 175, 175)),
            ('', (0, 0, 0)),
            (status.get('separation_text', ''), (230, 230, 230)),
            ('', (0, 0, 0)),
            ('GPS ellipse: reported 95%', (175, 175, 175)),
            ('No GPS heading at rest', (175, 175, 175)),
            ('Antenna offset unmeasured', (175, 175, 175)),
            ('Map/GPS are not ground truth', (175, 175, 175)),
            ('', (0, 0, 0)),
            ('q / Esc: close this view', (175, 175, 175)),
        ]
        for text, color in lines:
            cv2.putText(image, str(text)[:42], (left, y), cv2.FONT_HERSHEY_SIMPLEX, .49, color, 1, cv2.LINE_AA)
            y += 29
        return image


def timing_summary(values):
    if not values:
        return {'samples': 0}
    values = np.asarray(values, float)
    return {'samples': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'maximum_ms': float(values.max())}


def display_number(value, decimals):
    """Keep malformed/nonfinite diagnostics out of the visible numeric fields."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return 'unknown'
    return f'{value:.{decimals}f}' if math.isfinite(value) else 'unknown'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--alignment-dir', type=Path, required=True)
    result.add_argument('--gps-topic', default=GPS_TOPIC)
    result.add_argument('--pose-topic', default=POSE_TOPIC)
    result.add_argument('--localizer-status-topic', default=LOCALIZER_STATUS_TOPIC)
    result.add_argument('--width', type=int, default=1400)
    result.add_argument('--height', type=int, default=900)
    result.add_argument('--map-points', type=int, default=350000)
    result.add_argument('--path-points', type=int, default=4000)
    result.add_argument('--rate', type=float, default=10.)
    result.add_argument('--max-source-age', type=float, default=1.)
    result.add_argument('--max-receive-age', type=float, default=2.)
    result.add_argument('--no-display', action='store_true')
    result.add_argument('--publish-image', action='store_true')
    result.add_argument('--snapshot', type=Path)
    result.add_argument('--snapshot-period', type=float, default=5.)
    result.add_argument('--report', type=Path)
    result.add_argument('--duration', type=float, help='Optional wall-time limit in seconds')
    return result


def main(argv=None):
    args, ros_args = parser().parse_known_args(argv)
    if (not 0 < args.rate <= 30 or args.map_points < 1 or args.path_points < 1 or
            args.max_source_age <= 0 or args.max_receive_age <= 0 or args.snapshot_period <= 0 or
            (args.duration is not None and args.duration <= 0)):
        raise ValueError('display rates, point limits and age/duration limits must be positive')
    cv2.setNumThreads(1)
    start = time.monotonic()
    alignment, matrix, origin, pcd = load_map_alignment(args.alignment_dir)
    points = read_pcd_xyz(pcd)
    stride = max(1, math.ceil(len(points)/args.map_points))
    enu = points[::stride]@matrix[:3, :3].T+matrix[:3, 3]
    renderer = LocalizationMapRenderer(enu, args.width, args.height)
    startup_ms = (time.monotonic()-start)*1000

    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, NavSatFix
    from std_msgs.msg import String

    class LocalizationMapView(Node):
        def __init__(self):
            super().__init__('localization_map_view')
            self.source_mode = 'replay' if self.get_parameter('use_sim_time').value else 'live'
            self.gps, self.lidar = None, None
            self.gps_reason, self.lidar_reason = 'waiting for GPS', 'waiting for localizer'
            self.gps_path, self.lidar_path = deque(maxlen=args.path_points), deque(maxlen=args.path_points)
            self.localizer = {}
            self.localizer_received = 0.
            self.accepted_frames = {'enu', 'local_enu'}
            if alignment.get('enu_frame_id'):
                self.accepted_frames.add(alignment['enu_frame_id'])
            self.render_ms, self.gps_ms = deque(maxlen=20000), deque(maxlen=20000)
            self.counts = {'gps_received': 0, 'gps_invalid': 0, 'lidar_received': 0,
                           'lidar_invalid': 0, 'clock_rewinds': 0, 'frames': 0}
            self.previous_now, self.last_snapshot, self.last_status = 0, 0., 0.
            self.started = time.monotonic()
            self.quit = False
            self.last_summary = {}
            self.status_publisher = self.create_publisher(String, VIEW_STATUS_TOPIC, 2)
            self.image_publisher = self.create_publisher(Image, VIEW_IMAGE_TOPIC, 1) if args.publish_image else None
            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(NavSatFix, args.gps_topic, self.on_gps, qos)
            self.create_subscription(Odometry, args.pose_topic, self.on_pose, qos)
            self.create_subscription(String, args.localizer_status_topic, self.on_localizer, qos)
            # A ROS-time timer stops on paused playback, leaving a stale dot on
            # screen forever. Wall-clock rendering still evaluates ROS source age.
            self.create_timer(1/args.rate, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
            if not args.no_display:
                cv2.namedWindow('GPS vs LiDAR saved-map localization', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('GPS vs LiDAR saved-map localization', args.width, args.height)
            self.get_logger().info(f'North-up map ready; {len(enu):,} display points; no saved poses used')

        def on_gps(self, message):
            before = time.perf_counter()
            self.counts['gps_received'] += 1
            observation, reason = gps_observation(message, origin)
            if observation is None:
                self.counts['gps_invalid'] += 1
                self.gps, self.gps_reason = None, reason
            else:
                if self.gps is not None and observation['stamp_ns'] < self.gps['stamp_ns']:
                    self.gps_path.clear()
                observation['received_wall'] = time.monotonic()
                self.gps, self.gps_reason = observation, ''
                self.gps_path.append(observation['position'])
            self.gps_ms.append((time.perf_counter()-before)*1000)

        def on_pose(self, message):
            self.counts['lidar_received'] += 1
            observation, reason = lidar_observation(message, matrix, self.accepted_frames)
            if observation is None:
                self.counts['lidar_invalid'] += 1
                self.lidar, self.lidar_reason = None, reason
            else:
                if self.lidar is not None and observation['stamp_ns'] < self.lidar['stamp_ns']:
                    self.lidar_path.clear()
                observation['received_wall'] = time.monotonic()
                self.lidar, self.lidar_reason = observation, ''
                self.lidar_path.append(observation['position'])

        def on_localizer(self, message):
            try:
                value = json.loads(message.data)
                if isinstance(value, dict):
                    self.localizer = value
                    self.localizer_received = time.monotonic()
            except (ValueError, TypeError):
                pass

        def tick(self):
            wall_now, now = time.monotonic(), self.get_clock().now().nanoseconds
            if self.previous_now and now < self.previous_now-100_000_000:
                self.counts['clock_rewinds'] += 1
                self.gps, self.lidar = None, None
                self.gps_path.clear()
                self.lidar_path.clear()
                self.localizer = {}
            self.previous_now = now
            gps_fresh, gps_reason, gps_age = freshness(self.gps, now, wall_now, args.max_source_age, args.max_receive_age)
            lidar_fresh, lidar_reason, lidar_age = freshness(self.lidar, now, wall_now, args.max_source_age, args.max_receive_age)
            gps, lidar = self.gps if gps_fresh else None, self.lidar if lidar_fresh else None
            localizer = self.localizer if wall_now-self.localizer_received <= args.max_receive_age else {}
            state = str(localizer.get('state', localizer.get('status', 'waiting')))
            # Rejected/waiting scans do not keep an old accepted dot looking live.
            if (localizer.get('accepted') is False or
                    state.lower() in ('rejected', 'failed', 'lost', 'initializing', 'bootstrap', 'waiting', 'waiting_for_gps')):
                lidar = None
                lidar_reason = state
            info = {'gps_state': 'fresh '+{0: 'standalone', 1: 'SBAS', 2: 'GBAS'}.get(gps['fix_status'], 'fix') if gps else (self.gps_reason or gps_reason),
                    'lidar_state': state if lidar is not None else (self.lidar_reason or lidar_reason),
                    'gps_age_text': f'Source age: {gps_age:.3f} s' if gps_age is not None else '',
                    'lidar_age_text': f'Source age: {lidar_age:.3f} s' if lidar_age is not None else '',
                    'localizer_text': f"Compute: {display_number(localizer.get('compute_ms', localizer.get('latency_ms')), 1)} ms",
                    'source_mode': self.source_mode}
            sigma = None
            if gps is not None:
                if gps['covariance'] is not None:
                    sigma = float(np.sqrt(max(0., np.linalg.eigvalsh(gps['covariance'][:2, :2]).max())))
                    info['gps_sigma_text'] = f'Reported sigma max: {sigma:.2f} m'
                else:
                    info['gps_sigma_text'] = 'Reported covariance unknown'
            distance = None
            if gps is not None and lidar is not None and abs(gps['stamp_ns']-lidar['stamp_ns']) <= 150_000_000:
                distance = float(np.linalg.norm(gps['position'][:2]-lidar['position'][:2]))
                info['separation_text'] = f'GPS / LiDAR gap: {distance:.2f} m'
            if 'fitness' in localizer:
                info['matching_text'] = f"Match fitness: {display_number(localizer['fitness'], 3)}"
            before = time.perf_counter()
            image = renderer.render(gps, lidar, self.gps_path, self.lidar_path, info)
            self.render_ms.append((time.perf_counter()-before)*1000)
            self.counts['frames'] += 1
            self.last_summary = {'source_mode': self.source_mode,
                                 'gps_fresh': gps is not None, 'lidar_fresh': lidar is not None,
                                 'gps_state': info['gps_state'], 'lidar_state': info['lidar_state'],
                                 'gps_source_age_s': gps_age, 'lidar_source_age_s': lidar_age,
                                 'gps_reported_sigma_m': sigma, 'synchronized_position_gap_m': distance,
                                 'localizer': localizer, 'gap_is_accuracy_truth': False,
                                 'antenna_lever_arm_measured': False,
                                 'gps_processing': timing_summary(self.gps_ms),
                                 'rendering': timing_summary(self.render_ms), 'counts': self.counts.copy()}
            if self.image_publisher:
                message = Image()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = 'enu'
                message.height, message.width = image.shape[:2]
                message.encoding, message.step = 'bgr8', image.shape[1]*3
                message.data = image.tobytes()
                self.image_publisher.publish(message)
            if wall_now-self.last_status >= .5:
                self.status_publisher.publish(String(data=json.dumps(self.last_summary)))
                self.last_status = wall_now
            if args.snapshot and wall_now-self.last_snapshot >= args.snapshot_period:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.snapshot), image)
                self.last_snapshot = wall_now
            if not args.no_display:
                cv2.imshow('GPS vs LiDAR saved-map localization', image)
                if cv2.waitKey(1)&255 in (ord('q'), 27):
                    self.quit = True
            if args.duration and wall_now-self.started >= args.duration:
                self.quit = True

        def save_report(self):
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                report = {'startup_ms': startup_ms, 'display_map_points': len(enu),
                          'source_map_points': len(points), 'wall_duration_s': time.monotonic()-self.started,
                          'alignment': str(args.alignment_dir.resolve()), **self.last_summary}
                args.report.write_text(json.dumps(report, indent=2)+'\n')

    rclpy.init(args=ros_args)
    node = LocalizationMapView()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
