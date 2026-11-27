#!/usr/bin/env python3
"""Supervise live localized BEV and the bridge; never publish engagement."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from bev_dbw_bridge import parse_args as bridge_args

ROOT=Path(__file__).resolve().parent
BEV=ROOT.parent/'camera_lidar_bev'

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='default: live sensors, bridge serial disabled')
    mode.add_argument('--actuate',action='store_true',help='hardware connection with verified config; no auto-engagement')
    parser.add_argument('--hardware-config',type=Path)
    parser.add_argument('--map',type=Path,default=BEV/'maps/local_gps_recording_002')
    parser.add_argument('--max-speed-mph',type=float,default=6.,help='speed ceiling in mph (default: 6; maximum: 10)')
    parser.add_argument('--duration',type=float,help='optional positive wall-duration limit')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--print-only',action='store_true',help='validate and print commands, start/write nothing')
    args=parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0):
        parser.error('duration must be positive and finite')
    options=['--actuate' if args.actuate else '--dry-run','--max-speed-mph',str(args.max_speed_mph)]
    if args.hardware_config:options+=['--hardware-config',str(args.hardware_config.resolve())]
    # Fail before opening hardware, starting sensors or creating session files.
    bridge_args(options)
    output=(args.output or ROOT/'runs'/('all_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
    if output.exists():parser.error('choose a new output directory')
    if not (args.map/'gps_alignment/trajectory_enu.csv').is_file():parser.error('map has no ENU reference trajectory')
    engage='/bev_dbw/hardware_engage' if args.actuate else '/bev_dbw/dry_run_engage'
    commands={
        'view':['bash',str(BEV/'tools/run_localized_bev.sh'),'--map',str(args.map.resolve()),
                '--pure-pursuit','--range-m','80','--output',str(output/'view')],
        'bridge':['/usr/bin/python3',str(ROOT/'bev_dbw_bridge.py'),*options,
                  '--engage-topic',engage,'--log-dir',str(output/'bridge')],
    }
    if args.print_only:
        for name,command in commands.items():print(name+': '+shlex.join(command))
        print('No engagement publisher. No processes started or files written.')
        return 0
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(dict(commands=commands,actuate=args.actuate,
        engagement_published=False,hardware_config=str(args.hardware_config) if args.hardware_config else None),indent=2)+'\n')
    children=[];stopped=False;exit_code=0
    def request_stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    try:
        for name,command in commands.items():
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((name,process))
        print(f'Logs: {output}\nMode: {"HARDWARE (not engaged)" if args.actuate else "DRY RUN"}\n'
              f'Engagement topic: {engage}; no engagement is published by this launcher.\n'
              'Ctrl+C stops owned processes. Inspect /bev_dbw/bridge_status for readiness.',flush=True)
        started=time.monotonic()
        while not stopped:
            if args.duration and time.monotonic()-started>=args.duration:break
            exited=next(((name,p) for name,p in children if p.poll() is not None),None)
            if exited:
                name,p=exited;exit_code=int(p.returncode!=0)
                print(f'{name} exited ({p.returncode}); stopping the owned stack.',flush=True)
                break
            time.sleep(.1)
    finally:
        # Stop bridge first; let its bounded stop-intent period finish, then view.
        for name,process in reversed(children):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGINT)
                except ProcessLookupError:pass
            try:process.wait(timeout=40 if name=='view' else 8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        (output/'exit.json').write_text(json.dumps(dict(exit_code=exit_code,owned_children={name:p.poll() for name,p in children}),indent=2)+'\n')
    return exit_code

if __name__=='__main__':raise SystemExit(main())
