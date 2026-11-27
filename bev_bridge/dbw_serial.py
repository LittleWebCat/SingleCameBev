#!/usr/bin/env python3
"""Explicit serial transport; no discovery, retries, or shutdown commands.

Expected existing adapter line: 'steer_deg speed_mph pedal mode\n'.
The actual adapter firmware protocol must be verified before actuation.
"""
import math
import os
from pathlib import Path
import time
import serial


class DBW:
    def __init__(self, port, baud=115200, settle=2.0):
        if not port or not Path(port).exists():
            raise ValueError('provide an existing explicit DBW device path')
        identities=[str(port)]
        for link in Path('/dev/serial/by-id').glob('*'):
            if os.path.realpath(link)==os.path.realpath(port):identities.append(str(link))
        if any(word in ' '.join(identities).lower() for word in ('septentrio','ublox','u-blox','xsens')):
            raise ValueError('refusing GPS/IMU serial device')
        resolved=os.path.realpath(port)
        for process in Path('/proc').glob('[0-9]*'):
            if int(process.name)==os.getpid():continue
            try:
                for descriptor in (process/'fd').glob('*'):
                    if os.path.realpath(descriptor)==resolved:
                        raise ValueError(f'adapter already open by PID {process.name}')
            except (PermissionError,FileNotFoundError,ProcessLookupError):pass
        self.port=str(port)
        self.ser=serial.Serial(port=None,baudrate=baud,timeout=0,write_timeout=.02,exclusive=True)
        self.ser.dtr=False
        self.ser.rts=False
        self.ser.port=self.port
        self.ser.open()
        time.sleep(settle)
        self.last_deg=0.
        self.last_mode=0
        self.moving=False
        self.failed=False

    def _send(self,line,deadline_mono):
        if self.failed or self.ser is None or time.monotonic()>=deadline_mono:return False
        try:
            self.ser.write_timeout=min(.02,max(.001,deadline_mono-time.monotonic()))
            count=self.ser.write(line)
            if count!=len(line) or time.monotonic()>=deadline_mono:
                raise OSError('partial or expired serial write')
            return True
        except (serial.SerialException,OSError):
            self.failed=True
            self.close()
            return False

    def command(self,deg,speed_mph=0.,pedal=0.,mode=0,deadline_mono=None):
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in (deg,speed_mph,pedal)) or type(mode) is not int or mode not in (0,1)):
            raise ValueError('invalid serial command')
        if abs(deg)>640 or not 0<=speed_mph<=10 or abs(pedal)>1:
            raise ValueError('serial command exceeds software bounds')
        if (mode==0 and pedal!=0) or (mode==1 and speed_mph!=0):
            raise ValueError('mixed longitudinal command modes')
        deadline_mono=time.monotonic()+.02 if deadline_mono is None else deadline_mono
        if not math.isfinite(deadline_mono):raise ValueError('invalid write deadline')
        line=f'{deg:.2f} {speed_mph:.2f} {pedal:.4f} {mode}\n'.encode('ascii')
        if not self._send(line,deadline_mono):return False
        self.last_deg,self.last_mode=deg,mode
        self.moving |= speed_mph>0 or pedal>0
        return True

    def drain(self):
        if self.failed or self.ser is None:return ''
        try:return self.ser.read(4096).decode('ascii',errors='replace')
        except (serial.SerialException,OSError):
            self.failed=True
            self.close()
            return ''

    def close(self):
        # A never-engaged bridge must not emit steering/brake commands on exit.
        if self.ser is not None:
            try:self.ser.close()
            except (serial.SerialException,OSError):pass
            self.ser=None
