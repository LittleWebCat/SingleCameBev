#!/usr/bin/python3
"""Start/reuse raw sensors, snapshot their context, record a bounded test session."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from record_sensor_session import DEFAULT_TOPICS, positive_duration, storage_id
from start_bev import matching_processes, stop_owned


GPS_TOPICS = {
    '/sensing/gnss/fix': 'sensor_msgs/msg/NavSatFix',
    '/sensing/gnss/status': 'std_msgs/msg/String',
    '/sensing/gnss/raw_sbf': 'std_msgs/msg/UInt8MultiArray',
    '/sensing/gnss/diagnostics': 'diagnostic_msgs/msg/DiagnosticArray',
}


def capture_topics(gps=False):
    return list(DEFAULT_TOPICS)+(list(GPS_TOPICS) if gps else [])


def gps_observation(message, now_ns, received_at, monotonic_now,
                    allow_no_fix=False, receiver_status=None, status_age_s=None):
    """Require a recent real fix; the diagnostic override changes fix quality only."""
    source_ns = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
    source_age = (now_ns-source_ns)/1e9
    receipt_age = monotonic_now-received_at
    if (source_ns <= 0 or not 0 <= source_age <= 2.0 or
            not 0 <= receipt_age <= 2.0):
        raise ValueError('GPS fix is stale or uses a different ROS clock domain')
    if not message.header.frame_id:
        raise ValueError('GPS fix has no antenna frame')
    latitude, longitude, altitude = message.latitude, message.longitude, message.altitude
    valid = (message.status.status >= 0 and
             all(math.isfinite(v) for v in (latitude, longitude, altitude)) and
             -90 <= latitude <= 90 and -180 <= longitude <= 180)
    if not valid and not allow_no_fix:
        raise ValueError('GPS has no usable position fix; allow an outdoor sky view and wait')
    finite = lambda value: float(value) if math.isfinite(value) else None
    covariance = [finite(value) for value in message.position_covariance]
    return {
        'observed_at_utc': utc_now(), 'topic': '/sensing/gnss/fix',
        'frame_id': message.header.frame_id, 'header_stamp_ns': source_ns,
        'source_age_s': source_age, 'receipt_age_s': receipt_age,
        'status': int(message.status.status), 'service': int(message.status.service),
        'latitude_deg': finite(latitude), 'longitude_deg': finite(longitude),
        'altitude_m': finite(altitude), 'valid_fix': valid,
        'allow_no_fix': bool(allow_no_fix),
        'position_covariance_type': int(message.position_covariance_type),
        'position_covariance_m2': covariance,
        'covariance_known': message.position_covariance_type != 0 and all(
            value is not None for value in covariance),
        'receiver_status': receiver_status, 'receiver_status_age_s': status_age_s,
        'timestamp_source': 'host_ros_receipt',
        'physical_time_synchronization_validated': False,
        'antenna_lever_arm_measured': False,
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def output_paths(output):
    return (output, Path(str(output)+'.capture'), Path(str(output)+'.session.json'),
            Path(str(output)+'.recorder.log'), Path(str(output)+'.qc.json'))


def write_manifest(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def check_space(output, floor_gib):
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < (floor_gib+1)*1024**3:
        raise RuntimeError(f'Need at least {floor_gib+1:g} GiB free to begin a capture')
    return free


def lidar_profile(root, mode):
    filename = 'ouster_2048x10.yaml' if mode == '2048x10' else 'ouster_baseline.yaml'
    return root/'config/sensors'/filename


def context_snapshot(root, folder, lidar_mode='1024x20', gps_config=None):
    sources = [root/'config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml',
               root/'config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml',
               lidar_profile(root,lidar_mode), root/'config/sensors/front_camera.yaml',
               root/'config/transport/cyclonedds_large_clouds.xml', root/'config/recording/raw_sensor_qos.yaml']
    if gps_config is not None:
        sources.append(gps_config)
    snapshots = []
    for source in sources:
        shutil.copy2(source, folder/source.name)
        snapshots.append({'source': str(source), 'snapshot': source.name,
                          'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    return snapshots


def wait_for_sensors(root, folder, timeout, children, lidar_mode='1024x20',
                     gps=False, allow_no_gps_fix=False):
    """Receive one real scan, image metadata and latched sensor metadata before recording."""
    import rclpy
    import yaml
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, NavSatFix, PointCloud2
    from std_msgs.msg import String
    from lifecycle_msgs.srv import GetState

    expected = yaml.safe_load((folder/'front_camera_1024x768.candidate.yaml').read_text())
    required = {'/points': 'sensor_msgs/msg/PointCloud2', '/imu': 'sensor_msgs/msg/Imu',
                '/metadata': 'std_msgs/msg/String', '/camera/front/image_raw': 'sensor_msgs/msg/Image',
                '/camera/front/camera_info': 'sensor_msgs/msg/CameraInfo'}
    if gps:
        required.update(GPS_TOPICS)
    rclpy.init()
    node = rclpy.create_node('camera_lidar_capture_preflight')
    received = {}
    subscriptions = []
    try:
        subscriptions.append(node.create_subscription(CameraInfo, '/camera/front/camera_info',
                                                      lambda m: received.update(info=m), qos_profile_sensor_data))
        cloud_sub = None
        if not gps:
            cloud_sub = node.create_subscription(PointCloud2, '/points',
                                                 lambda m: received.setdefault('cloud', m), qos_profile_sensor_data)
            subscriptions.append(cloud_sub)
        if gps:
            subscriptions.append(node.create_subscription(NavSatFix, '/sensing/gnss/fix',
                lambda m: received.update(gps_fix=(m, time.monotonic())), qos_profile_sensor_data))
            subscriptions.append(node.create_subscription(String, '/sensing/gnss/status',
                lambda m: received.update(gps_status=(m, time.monotonic())), qos_profile_sensor_data))
        metadata_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        subscriptions.append(node.create_subscription(String, '/metadata',
                                                      lambda m: received.update(metadata=m), metadata_qos))
        deadline = time.monotonic()+timeout
        next_gps_notice = 0.0
        gps_snapshot = None
        gps_reason = 'waiting for a GPS message'
        while time.monotonic()<deadline:
            for name, process in children:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited before recording; inspect {folder/name}.log')
            rclpy.spin_once(node, timeout_sec=.1)
            if gps:
                gps_snapshot = None
                if 'gps_fix' in received:
                    fix, receipt = received['gps_fix']
                    status, status_age = None, None
                    if 'gps_status' in received:
                        status_message, status_receipt = received['gps_status']
                        try:
                            candidate = json.loads(status_message.data)
                            status = candidate if isinstance(candidate, dict) else None
                        except (ValueError, TypeError):
                            pass
                        status_age = time.monotonic()-status_receipt
                    try:
                        gps_snapshot = gps_observation(fix, node.get_clock().now().nanoseconds,
                            receipt, time.monotonic(), allow_no_gps_fix, status, status_age)
                    except ValueError as error:
                        gps_reason = str(error)
                if gps_snapshot is None and time.monotonic() >= next_gps_notice:
                    print(f'Waiting for GPS: {gps_reason}. Capture has not started.', flush=True)
                    next_gps_notice = time.monotonic()+10
                # Acquire a current scan after GPS is ready, then retain the latest
                # scan until all preflight inputs agree. A first cached scan would
                # expire while waiting for the receiver to obtain a fix.
                if gps_snapshot is not None and cloud_sub is None:
                    cloud_sub = node.create_subscription(PointCloud2, '/points',
                        lambda m: received.update(cloud=m), qos_profile_sensor_data)
                    subscriptions.append(cloud_sub)
            if not gps and cloud_sub is not None and 'cloud' in received:
                node.destroy_subscription(cloud_sub)
                subscriptions.remove(cloud_sub)
                cloud_sub = None
            graph = dict(node.get_topic_names_and_types())
            advertised = all(typ in graph.get(topic, []) for topic, typ in required.items())
            publishers = all(len(node.get_publishers_info_by_topic(topic))==1 for topic in required)
            if (advertised and publishers and all(key in received for key in ('info', 'cloud', 'metadata'))
                    and (not gps or gps_snapshot is not None)):
                break
        else:
            if gps and gps_snapshot is None:
                raise RuntimeError(f'GPS preflight timed out: {gps_reason}. '
                                   'Use --allow-no-gps-fix only for an explicit diagnostic recording.')
            raise RuntimeError('Camera/LiDAR did not supply the required raw topics and messages before the startup timeout')
        info, cloud = received['info'], received['cloud']
        if (info.width, info.height)!=(1024,768) or info.header.frame_id!='camera_front_optical_frame':
            raise RuntimeError('Existing camera dimensions/frame differ from the native 1024x768 capture profile')
        if info.distortion_model!=expected['distortion_model'] or len(info.d)!=5:
            raise RuntimeError('CameraInfo distortion schema differs from the calibration snapshot')
        for actual, wanted in ((info.k, expected['camera_matrix']['data']),
                               (info.d, expected['distortion_coefficients']['data']),
                               (info.r, expected['rectification_matrix']['data']),
                               (info.p, expected['projection_matrix']['data'])):
            if len(actual)!=len(wanted) or any(not math.isfinite(a) or abs(a-b)>1e-7 for a,b in zip(actual,wanted)):
                raise RuntimeError('Existing CameraInfo differs from the selected calibration snapshot')
        columns = int(lidar_mode.split('x')[0])
        if cloud.header.frame_id!='os_lidar' or (cloud.width,cloud.height)!=(columns,128):
            raise RuntimeError(f'Raw LiDAR is not os_lidar in the requested {lidar_mode} mode; '
                               'stop an existing driver in a different mode with Ctrl+C before collecting')
        fields = {f.name:f for f in cloud.fields}
        if any(name not in fields or fields[name].datatype!=7 for name in ('x','y','z')) or 't' not in fields or fields['t'].datatype!=6:
            raise RuntimeError('LiDAR must preserve original XYZ and UINT32 per-point t fields')
        if cloud.row_step!=cloud.width*cloud.point_step or len(cloud.data)!=cloud.row_step*cloud.height:
            raise RuntimeError('Invalid original LiDAR cloud payload')
        now_ns = node.get_clock().now().nanoseconds
        for msg in (info, cloud):
            source_ns = msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
            if not 0 <= now_ns-source_ns <= 2_000_000_000:
                raise RuntimeError('Sensor headers are not fresh in the current ROS clock domain')
        state_client = node.create_client(GetState, '/os_driver/get_state')
        if not state_client.wait_for_service(timeout_sec=2):
            raise RuntimeError('Ouster lifecycle service is missing')
        future = state_client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        if not future.done() or future.result() is None or future.result().current_state.id!=3:
            raise RuntimeError('The Ouster driver is not in the active lifecycle state')
        metadata = json.loads(received['metadata'].data)
        observed_mode = metadata.get('config_params',{}).get('lidar_mode')
        if observed_mode != lidar_mode:
            raise RuntimeError(f'LiDAR metadata mode {observed_mode!r} differs from requested {lidar_mode}; '
                               'stop the existing LiDAR driver with Ctrl+C before collecting')
        (folder/'ouster_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        observation = {'observed_at_utc': utc_now(), 'required_topics': required,
                       'camera_frame': info.header.frame_id, 'camera_size': [info.width,info.height],
                       'camera_k': list(info.k), 'camera_d': list(info.d),
                       'lidar_frame': cloud.header.frame_id, 'lidar_size': [cloud.width,cloud.height],
                       'lidar_mode': observed_mode,
                       'lidar_point_step': cloud.point_step,
                       'lidar_fields': [{'name': f.name,'offset': f.offset,'datatype': f.datatype,'count': f.count} for f in cloud.fields],
                       'physical_time_synchronization_validated': False}
        if gps:
            # Lifecycle checks can take seconds; require the latest GPS sample to
            # remain fresh immediately before recording is allowed to begin.
            rclpy.spin_once(node, timeout_sec=.1)
            fix, receipt = received['gps_fix']
            gps_snapshot = gps_observation(fix, node.get_clock().now().nanoseconds,
                receipt, time.monotonic(), allow_no_gps_fix,
                gps_snapshot['receiver_status'], gps_snapshot['receiver_status_age_s'])
            (folder/'gps_preflight.json').write_text(json.dumps(gps_snapshot, indent=2)+'\n')
            observation['gps'] = gps_snapshot
        (folder/'preflight.json').write_text(json.dumps(observation,indent=2)+'\n')
        return observation
    finally:
        node.destroy_node()
        rclpy.shutdown()


def snapshot_parameters(folder, lidar_mode='1024x20', gps=False):
    import yaml
    combined = {}
    nodes = [('/camera_lidar_bev/front_camera_bridge','camera_runtime.yaml'),('/os_driver','lidar_runtime.yaml')]
    if gps:
        nodes.append(('/septentrio_gps_bridge', 'gps_runtime.yaml'))
    for name, filename in nodes:
        result = subprocess.run(['ros2','param','dump',name],capture_output=True,text=True,timeout=20,check=True)
        (folder/filename).write_text(result.stdout)
        parameters = yaml.safe_load(result.stdout)
        if not isinstance(parameters,dict):
            raise RuntimeError(f'Cannot snapshot parameters for {name}')
        combined.update(parameters)
    (folder/'sensor_params.yaml').write_text(yaml.safe_dump(combined,sort_keys=False))
    camera = next((value['ros__parameters'] for key,value in combined.items() if key.endswith('front_camera_bridge')),None)
    lidar = next((value['ros__parameters'] for key,value in combined.items() if key.endswith('os_driver')),None)
    if not camera or camera.get('source_mode')!='camera' or camera.get('width')!=1024 or camera.get('height')!=768 or camera.get('fps')!=30.0:
        raise RuntimeError('Camera bridge must use physical native 1024x768 at the configured 30 FPS')
    if not lidar or lidar.get('lidar_mode')!=lidar_mode or lidar.get('point_type')!='original':
        raise RuntimeError(f'Active LiDAR parameters differ from the requested original {lidar_mode} profile; '
                           'stop the existing LiDAR driver with Ctrl+C before collecting')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=positive_duration,default=300.0)
    parser.add_argument('--lidar-mode',choices=('1024x20','2048x10'),default='1024x20',
                        help='Ouster mode: 1024x20 (default) or denser 2048x10 at 10 Hz')
    parser.add_argument('--scene',default='multi_scene_test',help='free-text scene label saved with the session')
    parser.add_argument('--gps', action='store_true',
                        help='include passive Septentrio GPS and require a fresh usable position fix')
    parser.add_argument('--gps-config', type=Path, default=root/'config/sensors/septentrio_gps.yaml',
                        help='passive GPS bridge configuration, copied into the capture context')
    parser.add_argument('--allow-no-gps-fix', action='store_true',
                        help='with --gps only: diagnostic capture of fresh GPS messages without a usable fix')
    parser.add_argument('--output',type=Path,help='new bag directory, optionally on another storage drive')
    parser.add_argument('--startup-timeout',type=positive_duration,default=60)
    parser.add_argument('--min-free-gib',type=float,default=2.0)
    parser.add_argument('--cache-mib',type=int,default=1024,
                        help='recorder producer-buffer MiB (default 1024; rosbag2 uses two buffers)')
    parser.add_argument('--dry-run',action='store_true',help='show commands without ROS nodes, sensors or file writes')
    args = parser.parse_args()
    if args.allow_no_gps_fix and not args.gps:
        parser.error('--allow-no-gps-fix requires --gps')
    gps_config = args.gps_config.expanduser().resolve()
    topics = capture_topics(args.gps)
    if not math.isfinite(args.min_free_gib) or args.min_free_gib<1:
        parser.error('minimum free-space floor must be at least 1 GiB')
    if not 1<=args.cache_mib<=4096:
        parser.error('recorder cache must be between 1 and 4096 MiB')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output or root/'bags'/f'camera_lidar_{timestamp}').expanduser().absolute()
    folder = Path(str(output)+'.capture')
    if any(p.exists() or p.is_symlink() for p in output_paths(output)):
        parser.error('refusing to overwrite an existing bag or capture/report path')
    calibration = root/'config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml'
    profile = lidar_profile(root,args.lidar_mode)
    commands = [
        ('camera_bridge',['ros2','launch','bev_bringup','front_camera.launch.py','width:=1024','height:=768',
                          'fps:=30.0',f'calibration_file:={calibration}']),
        ('os_driver',['/usr/bin/python3',str(root/'tools/start_lidar.py'),
                      '--params-file',str(folder/profile.name),'--startup-timeout','45']),
    ]
    if args.gps:
        commands.insert(0, ('gps_bridge', ['/usr/bin/python3', str(root/'tools/septentrio_gps.py'),
                                         '--config', str(folder/gps_config.name)]))
    recorder = ['/usr/bin/python3',str(root/'tools/record_sensor_session.py'),'--output',str(output),
                '--duration',str(args.duration),'--storage-preset-profile','zstd_fast',
                '--qos-overrides',str(root/'config/recording/raw_sensor_qos.yaml'),
                '--max-bag-size',str(2*1024**3),'--max-cache-size',str(args.cache_mib*1024**2),
                '--min-free-gib',str(args.min_free_gib), '--topics', *topics]
    quality = ['/usr/bin/python3',str(root/'tools/check_camera_lidar_bag.py'),str(output),
               '--calibration',str(folder/calibration.name),'--output',str(folder/'quality.json'),
               '--lidar-hz',str(int(args.lidar_mode.split('x')[1]))]
    print(f'Bag: {output}\nContext: {folder}\nDuration: {args.duration:g} s\nScene: {args.scene}\n'
          f'LiDAR mode: {args.lidar_mode}',flush=True)
    print(f'Storage: {storage_id()} (lossless zstd_fast for MCAP); free-space floor {args.min_free_gib:g} GiB',flush=True)
    if args.gps:
        print('GPS: passive receiver stream; '+('fresh diagnostic messages allowed without a fix'
              if args.allow_no_gps_fix else 'a fresh usable position fix is required')+
              '. No receiver commands or clock correction are sent.', flush=True)
    if args.dry_run:
        for _,command in commands:print(shlex.join(command))
        print(shlex.join(recorder));print(shlex.join(quality))
        print('Dry run: no sensors, ROS nodes, recorder, or files started.')
        return 0
    children, logs = [], []
    manifest = {'started_utc':utc_now(),'scene':args.scene,'requested_duration_s':args.duration,
                'bag_path':str(output),'capture_context':str(folder),'topics':topics,
                'ros_domain_id':os.environ.get('ROS_DOMAIN_ID','0'),'status':'preflight',
                'clock_alignment':'unverified; camera receipt stamps and ROS-derived LiDAR stamps',
                'intrinsic_accuracy_validated':False,'extrinsic_accuracy_validated':False,
                'odometry_validated':False,'owned_processes':[],'reused_processes':[],
                'requested_sensor_profiles':{'camera':'1024x768 MJPG 30 FPS','lidar':f'{args.lidar_mode} original'}}
    manifest['recorder_cache_mib']=args.cache_mib
    manifest['gps'] = {'enabled': args.gps, 'allow_no_fix': args.allow_no_gps_fix,
                       'topics': list(GPS_TOPICS) if args.gps else [],
                       'timestamp_source': 'host_ros_receipt' if args.gps else None,
                       'raw_receiver_time': 'GNSS week/TOW retained in status and raw SBF' if args.gps else None,
                       'physical_time_synchronization_validated': False,
                       'antenna_lever_arm_measured': False}
    result = 2
    try:
        if shutil.which('ros2') is None:
            raise RuntimeError('Use tools/collect_camera_lidar.sh to source the ROS environments')
        for path in (root/'install/bev_sensor_bridge/lib/bev_sensor_bridge/camera_bridge',calibration,
                     root/'tools/check_camera_lidar_bag.py',root/'config/recording/raw_sensor_qos.yaml',profile):
            if not path.is_file():raise RuntimeError(f'Required capture component is missing: {path}')
        if args.gps:
            for path in (root/'tools/septentrio_gps.py', gps_config):
                if not path.is_file():
                    raise RuntimeError(f'Required GPS capture component is missing: {path}')
        domain = os.environ.get('ROS_DOMAIN_ID','0')
        if not domain.isdecimal() or not 0<=int(domain)<=231:
            raise RuntimeError('Invalid ROS_DOMAIN_ID')
        try:buffer_limit=int(Path('/proc/sys/net/core/rmem_max').read_text())
        except (OSError,ValueError) as error:raise RuntimeError('Run this collector on the sensor host to inspect DDS buffers') from error
        if buffer_limit<16777216:
            raise RuntimeError('DDS receive buffer reset after reboot. Run:\nsudo sysctl -w net.core.rmem_max=16777216 net.core.rmem_default=4194304 net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728')
        manifest['initial_free_bytes']=check_space(output,args.min_free_gib)
        print(f'Available space: {manifest["initial_free_bytes"]/1024**3:.1f} GiB',flush=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        folder.mkdir()
        manifest['snapshots']=context_snapshot(root,folder,args.lidar_mode, gps_config if args.gps else None)
        git_result=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
        manifest['git_revision']=git_result.stdout.strip() if git_result.returncode==0 else None
        dirty=subprocess.run(['git','status','--porcelain','--','camera_lidar_bev'],cwd=root.parent,capture_output=True,text=True)
        manifest['project_worktree_status']=dirty.stdout
        (folder/'scene_notes.txt').write_text(f'Scene: {args.scene}\nRequested duration: {args.duration:g} seconds\n\nAdd elapsed times, object types, measured distances/dimensions, viewpoints and lighting here.\n')
        write_manifest(folder/'run_manifest.json',manifest)
        for name,command in commands:
            executable = 'septentrio_gps.py' if name == 'gps_bridge' else name
            existing=matching_processes(executable,domain)
            if existing:
                print(f'Reuse {name} PID {existing[0]} in domain {domain}',flush=True)
                manifest['reused_processes'].append({'name':name,'pid':existing[0]})
            else:
                log=(folder/f'{name}.log').open('x');logs.append(log)
                process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((name,process));manifest['owned_processes'].append({'name':name,'pid':process.pid})
                print(f'Start {name}; log {folder/name}.log',flush=True)
        print('Waiting for calibrated camera, original LiDAR scan and sensor metadata'+
              (' plus a fresh GPS fix...' if args.gps else '...'),flush=True)
        manifest['sensor_preflight']=wait_for_sensors(root,folder,args.startup_timeout,children,
            args.lidar_mode,args.gps,args.allow_no_gps_fix)
        snapshot_parameters(folder,args.lidar_mode,args.gps)
        manifest['status']='recording';manifest['recording_started_utc']=utc_now();write_manifest(folder/'run_manifest.json',manifest)
        print(f'Recording for {args.duration:g} seconds. Ctrl+C saves a partial bag and stops owned sensors.',flush=True)
        process=subprocess.Popen(recorder,cwd=root,start_new_session=True);children.append(('recorder_helper',process))
        result=process.wait()
        manifest['recorder_helper_returncode']=result
        manifest['status']='recorded' if result==0 else 'partial_or_failed'
    except KeyboardInterrupt:
        manifest['status']='interrupted';result=130
    except (OSError,RuntimeError,subprocess.SubprocessError,ValueError) as error:
        manifest['status']='failed';manifest['error']=str(error);print(f'Capture stopped: {error}',file=sys.stderr)
    finally:
        stop_owned(children)
        for log in logs:log.close()
        if folder.is_dir():
            manifest['sensor_cleanup_utc']=utc_now();write_manifest(folder/'run_manifest.json',manifest)
    if (output/'metadata.yaml').is_file() and result!=130:
        print('Checking saved raw messages offline. Sensors started here have stopped.',flush=True)
        try:
            checked=subprocess.run(quality,cwd=root)
            manifest['quality_checker_returncode']=checked.returncode
            if checked.returncode!=0 and result==0:result=2
        except (OSError,KeyboardInterrupt) as error:
            manifest['quality_check_error']=str(error);result=130 if isinstance(error,KeyboardInterrupt) else 2
    if folder.is_dir():
        manifest['ended_utc']=utc_now();write_manifest(folder/'run_manifest.json',manifest)
        print(f'Saved context: {folder}',flush=True)
        if (folder/'quality.json').is_file():
            print(f'Quality report: {folder/"quality.json"}',flush=True)
        elif 'recording_started_utc' not in manifest:
            print('Recording did not start; only configuration and startup logs were saved.',flush=True)
        else:
            print('No quality report was generated; inspect the session/recorder logs for the stop reason.',flush=True)
    return result


if __name__=='__main__':
    raise SystemExit(main())
