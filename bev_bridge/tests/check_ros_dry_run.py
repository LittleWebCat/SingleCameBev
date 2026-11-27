#!/usr/bin/python3
"""Isolated ROS bridge check. Dry-run children only; no hardware device access."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import rclpy
from std_msgs.msg import String,Bool

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='96':parser.error('use isolated ROS_DOMAIN_ID=96')
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init()
    node=rclpy.create_node('bridge_dry_test')
    targets=node.create_publisher(String,'/bridge_test/targets',10)
    engage=node.create_publisher(Bool,'/bridge_test/engage',10)
    statuses=[]
    subscription=node.create_subscription(String,'/bridge_test/status',lambda m:statuses.append(json.loads(m.data)),10)
    child=None;checks=0;case=0
    def check(value,reason):
        nonlocal checks
        if not value:raise AssertionError(reason)
        checks+=1
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.01)
    def stop():
        nonlocal child
        if child is not None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
            check(child.returncode==0,'owned dry-run child exits cleanly')
            child=None
    def start():
        nonlocal child,case
        stop();statuses.clear();case+=1
        folder=args.output/f'case_{case}'
        with (args.output/f'case_{case}.log').open('x') as log:
            child=subprocess.Popen(['/usr/bin/python3',str(ROOT/'bev_dbw_bridge.py'),'--dry-run',
                '--max-speed-mph','1',
                '--topic','/bridge_test/targets','--engage-topic','/bridge_test/engage',
                '--status-topic','/bridge_test/status','--log-dir',str(folder)],stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+5
        while not statuses and time.monotonic()<deadline:spin(.05)
        check(bool(statuses) and child.poll() is None,'bridge startup and DDS status')
    def sample(**changes):
        stamp=node.get_clock().now().nanoseconds
        fields=dict(frame='enu',pose_reference='rear_axle',source_mode='live_localization',
            diagnostic_only=True,tracking_valid=True,targets_valid=True,vehicle_transform_verified=True,
            state='tracking',source_stamp_ns=stamp,steering_rad=.1,curvature_per_m=.04,
            target_speed_mps=1.,estimated_speed_mps=1.)
        fields.update(changes)
        targets.publish(String(data=json.dumps(fields)))
    def feed(seconds,enabled=True,**changes):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            sample(**changes)
            if enabled is not None:engage.publish(Bool(data=enabled))
            spin(.05)
    def arm():
        feed(.15,False);feed(.3,True)
        check(any(s['ready'] for s in statuses),'fresh live input and explicit release/engage produce dry commands')
        check(statuses[-1]['last_output']['speed_mph']<=1.,'speed cap')
    try:
        spin(.3)
        check(not node.count_subscribers('/bridge_test/targets'),'test domain is not already occupied')
        start()
        feed(.3,True)
        check(not any(s['ready'] for s in statuses),'initial True publisher cannot auto-arm')
        arm()
        sample(tracking_valid=False,targets_valid=False,state='localization_stale');spin(.08)
        feed(.3,True)
        check(statuses[-1]['stop_latched'] and not statuses[-1]['ready'],'invalid input latches stop despite fresh later input')
        check(statuses[-1]['last_output']['speed_mph']==0.,'invalid input replaces positive target with stop intent')
        stop()
        start();arm()
        feed(.2,False)
        check(statuses[-1]['stop_latched'] and statuses[-1]['last_output']['speed_mph']==0.,'explicit False cancels driving')
        stop()
        start();arm()
        spin(.5)
        check(statuses[-1]['stop_latched'] and not statuses[-1]['ready'],'missing input/deadman latches stop')
        stop()
        start()
        feed(.3,True,source_mode='saved_recording_pose')
        check(not any(s['ready'] for s in statuses),'replay-labelled input rejected through actual ROS subscription')
        feed(.2,True,source_stamp_ns=node.get_clock().now().nanoseconds-5_000_000_000)
        check(statuses[-1]['last_output'] is None,'archived input never generates commands')
        targets.publish(String(data='[]'));spin(.15)
        check(child.poll() is None and not statuses[-1]['ready'],'malformed schema does not crash node')
        stop()
        events=[]
        for path in args.output.glob('case_*/events.jsonl'):
            events.extend(json.loads(line) for line in path.read_text().splitlines())
        command_events=[e for e in events if e['event']=='command']
        check(command_events and all(e['dry_run'] for e in command_events),'every generated command was dry-run')
        report=dict(checks_passed=checks,dry_run=True,hardware_ports_opened=False,
                    vehicle_commands_sent=False,ros_domain_id=96,owned_child_count=case,
                    diagnostic_command_count=len(command_events),physical_control_validated=False)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        if child is not None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
