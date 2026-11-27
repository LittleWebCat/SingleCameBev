#!/usr/bin/env python3
"""Live BEV bridge. Defaults to dry-run; hardware requires explicit configuration."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.clock import Clock,ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String,Bool

from bridge_core import InputGate,PedalPI,ResponseMonitor,StartGate,number
from dbw_serial import DBW

ROOT=Path(__file__).resolve().parent
MPS_TO_MPH=2.2369363
# dbw_serial.command rejects anything above this at the wire, independently of any CLI value.
# It is the backstop, and the reference for the overspeed trip when no CLI cap is set.
WIRE_MAX_MPH=10.


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='default: never open a serial port')
    mode.add_argument('--actuate',action='store_true',help='requires verified hardware configuration')
    parser.add_argument('--hardware-config',type=Path)
    parser.add_argument('--device',help='explicit device; must match hardware configuration')
    parser.add_argument('--topic',default='/camera_lidar_bev/path_tracking/status')
    parser.add_argument('--engage-topic',default='/bev_dbw/engage')
    parser.add_argument('--status-topic',default='/bev_dbw/bridge_status')
    parser.add_argument('--source',choices=('steering_rad','curvature'),default='steering_rad')
    parser.add_argument('--wheelbase',type=float,default=2.70)
    # Default: follow the planner. His pure_pursuit.yaml already caps itself at preview_speed_mps
    # 4.4704 m/s (= 10 mph) and lowers that for curvature and the endpoint, so passing his target
    # through faithfully is bounded by HIS limit. A CLI value is an extra, tighter operator cap.
    parser.add_argument('--max-speed-mph',type=float,default=None,
                        help='optional operator speed cap in mph (max 10); omitted = follow the planner')
    parser.add_argument('--lon-mode',choices=('speed_hold','pedal'),default='speed_hold')
    parser.add_argument('--rate',type=float,default=50.)
    parser.add_argument('--cmd-timeout',type=float,default=.3)
    parser.add_argument('--engage-timeout',type=float,default=.3)
    parser.add_argument('--source-timeout',type=float,default=.5)
    parser.add_argument('--max-road-angle',type=float,default=.5)
    parser.add_argument('--max-road-rate',type=float,default=.3)
    parser.add_argument('--kp',type=float,default=.15)
    parser.add_argument('--ki',type=float,default=.05)
    parser.add_argument('--max-pedal',type=float,default=.5)
    parser.add_argument('--stop-brake',type=float,default=.2)
    parser.add_argument('--log-dir',type=Path)
    # Closed-loop response checks against 0x741 telemetry (hardware mode only; see check_response).
    parser.add_argument('--no-response-checks',action='store_true',
                        help='disable the 0x741 measured-response trips (diagnostic use only)')
    parser.add_argument('--telemetry-timeout',type=float,default=1.,help='max age of 0x741 telemetry (s)')
    parser.add_argument('--overspeed-margin-mph',type=float,default=2.,help='measured mph over the cap before tripping')
    parser.add_argument('--no-motion-grace',type=float,default=3.,help='s of commanded speed with no measured motion before tripping')
    parser.add_argument('--steer-tol-deg',type=float,default=60.,help='measured-vs-commanded wheel error before tripping')
    parser.add_argument('--steer-grace',type=float,default=2.,help='s of steering error before tripping')
    # Start gate: the planner's rate limiter is skipped on its first tick after engage, so refuse
    # to start while it is saturated or badly off-path (see bridge_core.StartGate).
    parser.add_argument('--no-start-gate',action='store_true',help='disable the engage-time start checks')
    parser.add_argument('--max-engage-steer-deg',type=float,default=90.,help='opening demand allowed at engage')
    parser.add_argument('--max-engage-cte-m',type=float,default=1.5,help='cross-track allowed at engage')
    parser.add_argument('--max-engage-heading-rad',type=float,default=.35,help='heading error allowed at engage')
    parser.add_argument('--sat-grace',type=float,default=1.5,help='s of planner saturation before tripping')
    args=parser.parse_args(argv)
    try:
        for key in ('wheelbase','rate','cmd_timeout','engage_timeout','source_timeout','max_road_angle','max_road_rate'):
            if number(getattr(args,key))<=0:raise ValueError(key+' must be positive')
        for key in ('kp','ki'):
            if number(getattr(args,key))<0:raise ValueError(key+' must be nonnegative')
        if args.max_speed_mph is not None and number(args.max_speed_mph)<0:
            raise ValueError('max_speed_mph must be nonnegative')
        if not 10<=args.rate<=100:raise ValueError('rate must be 10..100 Hz')
        if args.max_speed_mph is not None and args.max_speed_mph>WIRE_MAX_MPH:
            raise ValueError('software speed cap is 10 mph')
        if args.max_road_angle>math.pi/2:raise ValueError('invalid road-wheel angle bound')
        for key in ('telemetry_timeout','overspeed_margin_mph','no_motion_grace','steer_tol_deg','steer_grace'):
            if number(getattr(args,key))<=0:raise ValueError(key+' must be positive')
        for key in ('max_engage_steer_deg','max_engage_cte_m','max_engage_heading_rad','sat_grace'):
            if number(getattr(args,key))<=0:raise ValueError(key+' must be positive')
        if args.max_engage_steer_deg>640:raise ValueError('engage steering limit must not exceed 640 degrees')
        if args.telemetry_timeout>2:raise ValueError('telemetry timeout must not exceed 2 seconds')
        if args.steer_tol_deg>180:raise ValueError('steering tolerance must not exceed 180 degrees')
        for key in ('max_pedal','stop_brake'):
            if not 0<number(getattr(args,key))<=1:raise ValueError(key+' must be in (0,1]')
        if not 0<args.cmd_timeout<=.5 or not 0<args.engage_timeout<=.5 or not 0<args.source_timeout<=.5:
            raise ValueError('timeouts must not exceed 0.5 seconds')
        args.dry_run=not args.actuate
        args.hardware=None
        if args.actuate:
            if not args.hardware_config:raise ValueError('--actuate requires --hardware-config')
            hardware=json.loads(args.hardware_config.read_text())
            if not isinstance(hardware,dict):raise ValueError('hardware config must be an object')
            external=hardware.get('safety_mode')=='operator_managed_external'
            if external:
                if not isinstance(hardware.get('external_safety_report'),str) or not hardware['external_safety_report'].strip():
                    raise ValueError('operator-managed mode requires the external safety responsibility report')
                if args.max_speed_mph is not None and args.max_speed_mph>WIRE_MAX_MPH:
                    raise ValueError('operator-managed speed cap is 10 mph')
            else:
                for key in ('protocol_verified','steering_calibration_verified','speed_feedback_verified',
                            'zero_speed_braking_verified','timeout_braking_verified','hardware_estop_verified'):
                    if hardware.get(key) is not True:raise ValueError('hardware verification missing: '+key)
            if hardware.get('protocol')!='steer_deg speed_mph pedal mode':raise ValueError('unsupported serial protocol')
            if type(hardware.get('baud')) is not int or not 1200<=hardware['baud']<=2_000_000:
                raise ValueError('provide the verified adapter baud rate')
            if type(hardware.get('steer_sign')) is not int or hardware['steer_sign'] not in (-1,1):
                raise ValueError('steer_sign must be -1 or 1')
            if not 0<number(hardware.get('steer_ratio'))<=100:raise ValueError('invalid steering ratio')
            if not 0<number(hardware.get('max_steering_wheel_deg'))<=640:raise ValueError('invalid wheel angle limit')
            if not 0<number(hardware.get('max_steering_wheel_rate_deg_s'))<=1000:raise ValueError('invalid wheel rate limit')
            if external:
                if hardware['max_steering_wheel_deg']>90 or hardware['max_steering_wheel_rate_deg_s']>90:
                    raise ValueError('operator-managed initial steering angle/rate caps are 90 deg and 90 deg/s')
            elif not 0<number(hardware.get('firmware_heartbeat_timeout_s'))<=1:
                raise ValueError('invalid firmware watchdog timeout')
            if not isinstance(hardware.get('verification_notes'),str) or not hardware['verification_notes'].strip():
                raise ValueError('document hardware verification evidence')
            device=hardware.get('device')
            if not isinstance(device,str) or not device.startswith('/dev/serial/by-id/'):
                raise ValueError('hardware device must be an explicit /dev/serial/by-id path')
            if args.device is not None and args.device!=device:raise ValueError('--device differs from verified configuration')
            args.device=device
            if args.lon_mode!='speed_hold':raise ValueError('hardware pedal mode requires a separate feedback interface; dry-run only')
            args.hardware=hardware
    except (ValueError,TypeError,OSError) as error:
        parser.error(str(error))
    return args


class BevDbwBridge(Node):
    # Class-level defaults so the measured-response state is always present, including on instances
    # built with object.__new__ in the tests. __init__ overwrites every one of them.
    telem=None
    telem_seen=False
    start=None
    start_block=None
    seeded=False
    first_cmd_mono=None
    last_cmd_speed=0.
    monitor=None

    def __init__(self,args):
        super().__init__('bev_dbw_bridge')
        self.a=args
        self.dbw=None
        self.gate=InputGate(args.cmd_timeout,args.engage_timeout,args.source_timeout,args.source,args.wheelbase)
        self.pi=PedalPI(args.kp,args.ki,args.max_pedal,args.stop_brake)
        # Uncapped means the planner governs, so the measured-overspeed trip references the wire
        # bound instead -- it must stay armed whether or not an operator cap is set.
        self.speed_ref=args.max_speed_mph if args.max_speed_mph is not None else WIRE_MAX_MPH
        self.monitor=ResponseMonitor(self.speed_ref,args.overspeed_margin_mph,args.telemetry_timeout,
                                     args.no_motion_grace,args.steer_tol_deg,args.steer_grace)
        self.start=None if args.no_start_gate else StartGate(
            args.max_engage_steer_deg,args.max_engage_cte_m,args.max_engage_heading_rad,args.sat_grace)
        self.start_block=None
        self.seeded=False
        self.last_tick=None
        self.last_sent_steer=0.
        self.sent_any=False
        self.stop_latched=False
        self.stop_reason=None
        self.last_target_mono=None
        self.last_engage_mono=None
        self.transport_fault=False
        self.last_status=0.
        self.last_dry_log=0.
        self.last_output=None
        self.log_failed=False
        self.telem=None              # newest 0x741 snapshot (None in dry-run)
        self.telem_seen=False
        self.first_cmd_mono=None
        self.last_cmd_speed=0.
        self.hardware=args.hardware or dict(steer_ratio=17.,steer_sign=-1,
            max_steering_wheel_deg=640.,max_steering_wheel_rate_deg_s=math.degrees(args.max_road_rate)*17.)
        if args.actuate and self.get_parameter('use_sim_time').value:
            raise ValueError('hardware mode cannot use simulated time')
        log_dir=args.log_dir or ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        log_dir.mkdir(parents=True,exist_ok=False)
        self.log_file=(log_dir/'events.jsonl').open('x',buffering=1)
        self.event('startup',dry_run=args.dry_run,device=args.device,hardware=args.hardware)
        if args.actuate:
            try:self.dbw=DBW(args.device,baud=self.hardware['baud'])
            except Exception:
                self.log_file.close()
                raise
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String,args.topic,self.on_status,qos)
        self.create_subscription(Bool,args.engage_topic,self.on_engage,qos)
        self.status_pub=self.create_publisher(String,args.status_topic,1)
        self.steady_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1/args.rate,self.tick,clock=self.steady_clock)
        self.get_logger().info(f'Mode={"DRY RUN" if args.dry_run else "HARDWARE"}; logs={log_dir}; '
                               'requires fresh live targets, then engage=False followed by fresh True heartbeats.')

    def event(self,name,**fields):
        if self.log_failed:return
        record=dict(event=name,wall_time_ns=time.time_ns(),monotonic_s=time.monotonic(),**fields)
        try:self.log_file.write(json.dumps(record,allow_nan=False)+'\n')
        except OSError:
            self.log_failed=True
            self.gate.invalidate('logging_fault_restart_required')
            if self.sent_any:self.latch_stop('log_write_failed')
            self.get_logger().error('Diagnostic log write failed; positive commands disabled.')

    def latch_stop(self,cause):
        # Record why the stop latched once, before later inputs overwrite gate.reason.
        if self.stop_latched:return
        self.stop_latched=True
        now=time.monotonic();ros_ns=self.get_clock().now().nanoseconds
        def age(mono):return None if mono is None else round(now-mono,4)
        self.stop_reason=dict(cause=cause,gate_reason=self.gate.reason,
            target_receipt_age_s=age(self.last_target_mono),
            source_age_s=round((ros_ns-self.gate.last_stamp_ns)/1e9,4) if self.gate.last_stamp_ns else None,
            engage_true_age_s=age(self.last_engage_mono),
            cmd_timeout_s=self.a.cmd_timeout,source_timeout_s=self.a.source_timeout,
            engage_timeout_s=self.a.engage_timeout,last_output=self.last_output,
            telemetry=getattr(self,'telem',None))
        self.event('stop_latched',**self.stop_reason)
        self.get_logger().warning('STOP LATCHED: '+json.dumps(self.stop_reason))

    def on_status(self,message):
        received=time.monotonic()
        accepted=self.gate.receive(message.data,received,self.get_clock().now().nanoseconds)
        if accepted:self.last_target_mono=received
        self.event('target',accepted=accepted,reason=self.gate.reason,
                   source_stamp_ns=self.gate.last_stamp_ns,
                   target=None if self.gate.target is None else vars(self.gate.target))
        if not accepted:
            self.pi.reset()
            if self.sent_any:self.latch_stop('target_rejected')

    def measured_deg(self,now):
        """Measured wheel angle, if telemetry is fresh and numeric. None otherwise -- a dry run has
        no board, and a half-built node in the tests has no monitor."""
        if self.dbw is None or self.monitor is None:return None
        deg=getattr(self.dbw,'meas_deg',None)
        if isinstance(deg,bool) or not isinstance(deg,(int,float)):return None
        age=self.dbw.meas_age(now)
        if isinstance(age,bool) or not isinstance(age,(int,float)):return None
        return float(deg) if age<=self.a.telemetry_timeout else None

    def wheel_deg(self,road_rad):
        """Planner road-wheel angle -> steering-wheel degrees, before the rate limit."""
        road=max(-self.a.max_road_angle,min(self.a.max_road_angle,road_rad))
        bound=self.hardware['max_steering_wheel_deg']
        return max(-bound,min(bound,self.hardware['steer_sign']*math.degrees(road)*self.hardware['steer_ratio']))

    def on_engage(self,message):
        # Gate the TRANSITION into engaged only; once driving, the saturation trip takes over.
        if (message.data is True and self.start is not None and self.gate.engage_rx is None
                and not self.gate.release_required):
            target=self.gate.target
            demand=self.wheel_deg(target.steering_rad) if target is not None else 0.
            blocked=self.start.blocks_engage(demand,target)
            if blocked:
                if blocked!=self.start_block:
                    self.get_logger().warning('ENGAGE REFUSED: '+blocked)
                    self.event('engage_refused',reason=blocked,demand_deg=round(demand,2))
                self.start_block=blocked
                return                       # never reaches gate.engage(): nothing arms
            self.start_block=None
        self.gate.engage(message.data,time.monotonic())
        if message.data is True:self.last_engage_mono=time.monotonic()
        if message.data is False:
            self.pi.reset()
            if self.sent_any:self.latch_stop('operator_disengaged')
        self.event('engage',value=message.data,reason=self.gate.reason)

    def write(self,command,deadline):
        self.last_output=command
        if self.a.dry_run:return True
        if self.dbw is None or self.transport_fault:return False
        try:ok=self.dbw.command(**command,deadline_mono=deadline)
        except Exception as error:
            self.event('write_exception',error=str(error));ok=False
        if not ok:
            self.transport_fault=True
            self.gate.invalidate('serial_fault_restart_required')
            self.latch_stop('serial_write_failed')
            self.pi.reset()
            self.dbw.close()
            self.get_logger().error('Serial write failed; latched fault, no reconnect/retry. Hardware watchdog/operator stop required.')
        return ok

    def check_response(self,now):
        """Hardware-mode only: compare 0x741 telemetry against what we commanded (ResponseMonitor)."""
        if self.monitor is None or self.dbw is None:return
        if self.a.dry_run or self.a.no_response_checks:return
        if not self.sent_any or self.stop_latched or self.transport_fault:return
        trip=self.monitor.update(now,self.dbw.meas_age(now),self.dbw.meas_mph,self.dbw.meas_deg,
                                 self.last_cmd_speed,self.last_sent_steer,self.first_cmd_mono)
        if trip:self.latch_stop(trip)

    def check_planner(self,now):
        """Runtime half of the start gate: a planner stuck at full lock is the cross-track abort
        case, and is a stop whether or not we have board telemetry."""
        if self.start is None or self.stop_latched or not self.sent_any:return
        trip=self.start.runtime(now,self.gate.target)
        if trip:self.latch_stop(trip)

    def stop_command(self,now):
        # Hold the last transmitted steering; no automatic centering during a stop.
        command=dict(deg=self.last_sent_steer,speed_mph=0.,pedal=0.,mode=0)
        self.last_cmd_speed=0.
        self.write(command,now+.02)
        self.event('stop_intent',command=command,transport_fault=self.transport_fault)

    def tick(self):
        now=time.monotonic();ros_ns=self.get_clock().now().nanoseconds
        dt=1/self.a.rate if self.last_tick is None else now-self.last_tick
        self.last_tick=now
        if self.dbw is not None and self.monitor is not None and not self.transport_fault:
            self.dbw.poll(now)
            self.telem=self.dbw.telemetry()
        self.check_response(now)
        self.check_planner(now)
        if self.get_parameter('use_sim_time').value and self.a.actuate:
            self.gate.clock_fault=True
        ready=self.gate.ready(now,ros_ns)
        if dt<=0 or dt>.1:
            ready=False;self.gate.invalidate('control_loop_delay_release_required')
        if self.transport_fault or self.log_failed:ready=False
        if self.sent_any and not ready:self.latch_stop('not_ready')
        if self.stop_latched:
            self.pi.reset()
            if self.sent_any and not self.transport_fault:self.stop_command(now)
            ready=False
        elif ready:
            target=self.gate.target
            # Bumpless start: slew from where the wheel actually IS, not from an assumed zero.
            measured=self.measured_deg(now)
            if not self.seeded and measured is not None:
                self.last_sent_steer=measured
                self.event('seeded_from_measured',deg=measured)
                self.seeded=True
            requested=self.wheel_deg(target.steering_rad)
            step=min(self.hardware['max_steering_wheel_rate_deg_s'],
                     math.degrees(self.a.max_road_rate)*self.hardware['steer_ratio'])*dt
            steering=max(self.last_sent_steer-step,min(self.last_sent_steer+step,requested))
            speed=target.speed_mps*MPS_TO_MPH          # follow the planner faithfully
            if self.a.max_speed_mph is not None:
                speed=min(self.a.max_speed_mph,speed)  # optional, tighter operator cap
            speed=min(speed,WIRE_MAX_MPH)              # never offer the wire more than it accepts
            command=dict(deg=steering,speed_mph=speed,pedal=0.,mode=0)
            if self.a.lon_mode=='pedal':
                command.update(speed_mph=0.,pedal=self.pi.update(speed/MPS_TO_MPH,target.feedback_mps,dt),mode=1)
            deadline=min(now+.02,target.received_mono+self.a.cmd_timeout,
                         self.gate.engage_rx+self.a.engage_timeout,
                         now+self.a.source_timeout-(ros_ns-target.stamp_ns)/1e9)
            if self.write(command,deadline):
                self.sent_any=True
                self.last_sent_steer=steering
                self.last_cmd_speed=speed
                if self.first_cmd_mono is None:self.first_cmd_mono=now
                self.event('command',source_stamp_ns=target.stamp_ns,command=command,dry_run=self.a.dry_run,
                           telemetry=self.telem)
                if self.a.dry_run and now-self.last_dry_log>=.2:
                    self.get_logger().info(f'[dry-run] steer={steering:+.2f}deg speed={speed:.2f}mph mode={command["mode"]}')
                    self.last_dry_log=now
        else:self.pi.reset()
        if now-self.last_status>=.1:
            state='serial_fault_restart_required' if self.transport_fault else 'logging_fault_restart_required' if self.log_failed else 'stop_latched_restart_required' if self.stop_latched else self.gate.reason
            status=dict(dry_run=self.a.dry_run,ready=ready and not self.transport_fault,
                        state=state,stop_latched=self.stop_latched,stop_reason=self.stop_reason,
                        transport_fault=self.transport_fault,
                        speed_cap_mph=self.a.max_speed_mph,speed_trip_ref_mph=self.speed_ref,
                        start_gate_block=self.start_block,
                        safety_mode=self.hardware.get('safety_mode','verified_configuration' if self.a.actuate else 'dry_run'),
                        firmware_verified=self.hardware.get('protocol_verified') is True and self.hardware.get('timeout_braking_verified') is True,
                        release_required=self.gate.release_required,source_stamp_ns=self.gate.last_stamp_ns,
                        last_output=self.last_output,telemetry=self.telem,
                        response_checks=not (self.a.dry_run or self.a.no_response_checks))
            self.status_pub.publish(String(data=json.dumps(status,allow_nan=False)))
            self.last_status=now

    def shutdown(self):
        # Best-effort stop intent, not a claim of standstill. Firmware behavior
        # and an operator brake/estop are required if serial cannot deliver it.
        try:
            if self.dbw is not None and self.sent_any and not self.transport_fault:
                deadline=time.monotonic()+4
                while time.monotonic()<deadline and not self.transport_fault:
                    self.stop_command(time.monotonic());time.sleep(.02)
        finally:
            if self.dbw is not None:self.dbw.close();self.dbw=None
            try:self.event('shutdown',sent_any=self.sent_any,transport_fault=self.transport_fault)
            finally:self.log_file.close()


def main(argv=None):
    args=parse_args(argv)
    rclpy.init(args=[])
    node=None
    try:
        node=BevDbwBridge(args)
        def stop(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        rclpy.spin(node)
    except KeyboardInterrupt:pass
    except (OSError,ValueError,RuntimeError) as error:
        print(f'Bridge stopped: {error}',file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if node is not None:
            node.shutdown();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0


if __name__=='__main__':sys.exit(main())
