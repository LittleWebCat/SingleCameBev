#!/usr/bin/python3
"""Isolated ROS software check; starts only an owned diagnostic tracker."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import yaml
import rclpy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from saved_map_source import read_trajectory, quaternion_rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output file')
    if os.environ.get('ROS_DOMAIN_ID')!='94':
        parser.error('run in isolated ROS_DOMAIN_ID=94')
    rclpy.init()
    node=rclpy.create_node('path_tracking_software_check')
    clock_pub=node.create_publisher(Clock,'/clock',10)
    pose_pub=node.create_publisher(Odometry,'/camera_lidar_bev/path_tracking/source_pose',10)
    records=[]
    subscription=node.create_subscription(String,'/camera_lidar_bev/path_tracking/status',
                                         lambda message: records.append(json.loads(message.data)),10)
    child=None
    checks=0
    processing=[]
    def spin(seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.01)
    def check(condition,reason):
        nonlocal checks
        if not condition:
            raise AssertionError(reason)
        checks+=1
    def stop():
        nonlocal child
        if child is not None:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate(); child.wait(timeout=3)
            child=None
    def start(csv_path,calibrated=False,synthetic=False):
        nonlocal child
        stop(); records.clear()
        command=[str(ROOT/'install/bev_path_tracking/lib/bev_path_tracking/path_tracker'),
            '--ros-args','--params-file',str(ROOT/'config/tracking/pure_pursuit.yaml'),
            '-p',f'reference_trajectory:={csv_path}','-p','use_sim_time:=true',
            '-p','source_mode:=saved_recording_pose']
        if calibrated is not None:
            command += ['-p',f'vehicle_transform_verified:={str(calibrated).lower()}']
        if synthetic:
            command += ['-p','lidar_forward_from_rear_axle_m:=0.0',
                        '-p','lidar_left_from_rear_axle_m:=0.0','-p','sensor_yaw_in_vehicle_rad:=0.0']
        child=subprocess.Popen(command,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        spin(1.)
        check(child.poll() is None,'tracker startup')
    def publish(stamp,xyz,quaternion,frame='enu'):
        start_index=len(records)
        clock=Clock()
        clock.clock.sec,clock.clock.nanosec=divmod(int(stamp)+20_000_000,10**9)
        clock_pub.publish(clock); spin(.035)
        pose=Odometry()
        pose.header.frame_id,pose.child_frame_id=frame,'os_sensor'
        pose.header.stamp.sec,pose.header.stamp.nanosec=divmod(int(stamp),10**9)
        p,q=pose.pose.pose.position,pose.pose.pose.orientation
        p.x,p.y,p.z=map(float,xyz)
        q.x,q.y,q.z,q.w=map(float,quaternion)
        pose_pub.publish(pose)
        deadline=time.monotonic()+1.
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.01)
            matches=[record for record in records[start_index:] if record.get('source_stamp_ns')==int(stamp)
                     and record.get('state')!='localization_stale']
            if matches:
                return matches[-1]
        raise AssertionError('missing source-stamped tracking result')
    try:
        reference=ROOT/'maps/local_gps_recording_002/gps_alignment/trajectory_enu.csv'
        stamps,positions,quaternions=read_trajectory(reference)
        start(reference)
        for index in range(600,630):
            result=publish(stamps[index],positions[index],quaternions[index])
            check(result['tracking_valid'],'recorded source pose matches reference')
            check(not result['targets_valid'] and result['steering_rad'] is None,'unset axle must withhold steering')
            check(not result['vehicle_curvature_valid'] and result['vehicle_curvature_per_m'] is None,
                  'unset axle must withhold vehicle curvature')
            check(result['diagnostic_only'] and result['frame']=='enu','diagnostic ENU contract')
            processing.append(result['processing_ms'])
        result=publish(stamps[500],positions[500],quaternions[500])
        check(result['tracking_valid'],'clock rewind reacquires correct route')
        spin(.7)
        check(records[-1]['state']=='localization_stale' and not records[-1]['tracking_valid'],
              'paused clock and stale wall receipt clear targets')
        check(records[-1]['vehicle_curvature_per_m'] is None,'stale localization clears vehicle curvature')
        before=len(records)
        clock=Clock(); clock.clock.sec,clock.clock.nanosec=divmod(int(stamps[501])+20_000_000,10**9)
        clock_pub.publish(clock); spin(.04)
        invalid=Odometry(); invalid.header.frame_id='map'; invalid.child_frame_id='os_sensor'
        invalid.header.stamp.sec,invalid.header.stamp.nanosec=divmod(int(stamps[501]),10**9)
        invalid.pose.pose.orientation.w=1.
        pose_pub.publish(invalid); spin(.1)
        check(any(r['state']=='invalid_pose_frame_or_stamp' for r in records[before:]),'wrong frame rejected')
        stop()
        parameters=yaml.safe_load((ROOT/'config/tracking/pure_pursuit.yaml').read_text())['path_tracker']['ros__parameters']
        configured_count=0
        if parameters['vehicle_transform_verified']:
            start(reference,None)
            for index in range(600,610):
                result=publish(stamps[index],positions[index],quaternions[index])
                check(result['tracking_valid'] and result['pose_reference']=='rear_axle',
                      'configured vehicle transform uses rear axle')
                rotation=quaternion_rotation(quaternions[index])
                yaw=math.atan2(rotation[1,0],rotation[0,0])-parameters['sensor_yaw_in_vehicle_rad']
                forward,left=parameters['lidar_forward_from_rear_axle_m'],parameters['lidar_left_from_rear_axle_m']
                expected=positions[index].copy()
                expected[:2]-=[math.cos(yaw)*forward-math.sin(yaw)*left,
                               math.sin(yaw)*forward+math.cos(yaw)*left]
                check(np.allclose(result['ego_xyz'],expected,atol=1e-5,rtol=0),
                      'configured rear axle translation and yaw applied')
                if index>600:
                    check(result['targets_valid'] and all(np.isfinite(result[key]) for key in
                          ('curvature_per_m','steering_rad','target_speed_mps')),
                          'configured consecutive poses produce finite preview targets')
                configured_count+=1
            stop()
        with tempfile.TemporaryDirectory(prefix='pure_pursuit_ros_') as folder:
            fixture=Path(folder)/'straight.csv'
            with fixture.open('w',newline='') as stream:
                writer=csv.writer(stream); writer.writerow(['stamp_sec','x_m','y_m','z_m','qx','qy','qz','qw'])
                for index in range(21): writer.writerow([1+index*.1,index,0,0,0,0,0,1])
            start(fixture,True,synthetic=True)
            publish(100_000_000_000,[1,1,0],[0,0,0,1])
            result=publish(100_100_000_000,[1.1,1,0],[0,0,0,1])
            check(result['targets_valid'] and result['steering_rad']<0,'synthetic calibrated left offset steers right')
            check(result['target_speed_mps']<=parameters['preview_speed_mps'],'preview speed limit')
            check(result['pose_reference']=='rear_axle','calibrated basis')
            check(not result['vehicle_curvature_valid'],'vehicle curvature requires a motion window')
            for index in range(2,10):
                result=publish(100_000_000_000+index*100_000_000,[1+.1*index,1,0],[0,0,0,1])
            check(result['vehicle_curvature_valid'] and abs(result['vehicle_curvature_per_m'])<1e-8,
                  'straight observed motion has zero curvature')
            check(result['curvature_per_m']<0 and result['vehicle_curvature_source']=='localization_pose_estimate',
                  'observed curvature differs from desired steering correction')
            result=publish(101_000_000_000,[1.9,1,0],[0,0,0,1])
            check(not result['vehicle_curvature_valid'] and result['vehicle_curvature_per_m'] is None,
                  'stopped vehicle leaves curvature unavailable')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(
            software_checks_passed=checks,reference_pose_count=len(stamps),sampled_recorded_poses=30,
            geometry_self_reference=True,physical_sensors_started=False,vehicle_commands_published=False,
            calibration_for_synthetic_case_only=configured_count==0,accuracy_independently_validated=False,
            configured_vehicle_pose_count=configured_count,
            core_processing_ms_median=float(np.median(processing)),
            core_processing_ms_p95=float(np.percentile(processing,95))),indent=2)+'\n')
        print(args.output.read_text())
    finally:
        stop()
        node.destroy_subscription(subscription)
        node.destroy_node(); rclpy.shutdown()


if __name__=='__main__':
    main()
