import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bridge_core import InputGate,PedalPI
from dbw_serial import DBW
from bev_dbw_bridge import parse_args,BevDbwBridge


def message(stamp=10_000_000_000,**changes):
    fields=dict(frame='enu',pose_reference='rear_axle',source_mode='live_localization',
        diagnostic_only=True,tracking_valid=True,targets_valid=True,vehicle_transform_verified=True,
        state='tracking',source_stamp_ns=stamp,steering_rad=.1,curvature_per_m=.04,
        target_speed_mps=1.,estimated_speed_mps=1.)
    fields.update(changes)
    return json.dumps(fields)


class GateTests(unittest.TestCase):
    def ready(self):
        gate=InputGate()
        self.assertTrue(gate.receive(message(),10.,10_000_000_000))
        gate.engage(False,10.);gate.engage(True,10.)
        self.assertTrue(gate.ready(10.,10_000_000_000))
        return gate

    def test_explicit_false_revokes_immediately(self):
        gate=self.ready();gate.engage(False,10.01)
        self.assertFalse(gate.ready(10.01,10_010_000_000))

    def test_initial_true_cannot_arm(self):
        gate=InputGate();gate.receive(message(),10.,10_000_000_000)
        gate.engage(True,10.)
        self.assertFalse(gate.ready(10.,10_000_000_000))

    def test_invalid_status_clears_previously_accepted_target(self):
        for change in ({'tracking_valid':False},{'targets_valid':False},{'state':'localization_stale'},
                       {'steering_rad':None},{'estimated_speed_mps':None}):
            gate=self.ready()
            self.assertFalse(gate.receive(message(10_010_000_000,**change),10.01,10_010_000_000))
            self.assertIsNone(gate.target)
            gate.engage(True,10.01)
            self.assertFalse(gate.ready(10.01,10_010_000_000))

    def test_bad_json_types_and_bool_numbers_rejected(self):
        for payload in ('[]','null','true','1','"text"','{',
                        '{"tracking_valid":true,"tracking_valid":false}',
                        message(steering_rad=True),message(tracking_valid='true'),
                        message(steering_rad=float('nan')),message(steering_rad=float('inf')),
                        message(source_stamp_ns=True),message(target_speed_mps=-1)):
            gate=self.ready()
            self.assertFalse(gate.receive(payload,10.01,10_010_000_000))
            self.assertIsNone(gate.target)

    def test_saved_replay_wrong_frames_and_transform_rejected(self):
        for change in ({'source_mode':'saved_recording_pose'},{'source_mode':'viewer_paired_pose'},
                       {'frame':'map'},{'pose_reference':'os_sensor'},{'vehicle_transform_verified':False}):
            gate=InputGate()
            self.assertFalse(gate.receive(message(**change),10.,10_000_000_000))

    def test_old_and_future_source_stamps_rejected(self):
        for stamp in (1,9_000_000_000,10_100_000_000):
            self.assertFalse(InputGate().receive(message(stamp),10.,10_000_000_000))

    def test_duplicates_and_out_of_order_invalidate(self):
        for stamp in (10_000_000_000,9_999_999_999):
            gate=self.ready()
            self.assertFalse(gate.receive(message(stamp),10.01,10_010_000_000))
            self.assertIsNone(gate.target)

    def test_timeouts_require_release_before_reengagement(self):
        gate=self.ready()
        self.assertFalse(gate.ready(10.31,10_310_000_000))
        gate.receive(message(10_320_000_000),10.32,10_320_000_000)
        gate.engage(True,10.32)
        self.assertFalse(gate.ready(10.32,10_320_000_000))
        gate.engage(False,10.32);gate.engage(True,10.32)
        self.assertTrue(gate.ready(10.32,10_320_000_000))

    def test_ros_clock_rewind_or_pause_latches_fault(self):
        for mono,ros_ns in ((10.01,1_000_000_000),(10.4,10_000_000_000),(10.01,11_000_000_000)):
            gate=self.ready()
            self.assertFalse(gate.ready(mono,ros_ns))
            self.assertTrue(gate.clock_fault)

    def test_endpoint_nonzero_speed_rejected(self):
        self.assertFalse(InputGate().receive(message(state='endpoint'),10.,10_000_000_000))
        self.assertTrue(InputGate().receive(message(state='endpoint',target_speed_mps=0.),10.,10_000_000_000))

    def test_raw_curvature_conversion(self):
        gate=InputGate(source='curvature')
        self.assertTrue(gate.receive(message(),10.,10_000_000_000))
        self.assertAlmostEqual(gate.target.steering_rad,.107582,places=5)


class SerialTests(unittest.TestCase):
    def link(self):
        link=object.__new__(DBW);link.ser=Mock();link.failed=False
        link.last_deg=0.;link.last_mode=0;link.moving=False
        link.ser.write.side_effect=lambda line:len(line)
        return link

    def test_command_format(self):
        link=self.link()
        with patch('dbw_serial.time.monotonic',return_value=10.):
            self.assertTrue(link.command(-2.,speed_mph=1.,deadline_mono=10.02))
        link.ser.write.assert_called_once_with(b'-2.00 1.00 0.0000 0\n')

    def test_ten_mph_boundary_reaches_transport_but_higher_speed_does_not(self):
        link=self.link()
        with patch('dbw_serial.time.monotonic',return_value=10.):
            self.assertTrue(link.command(0.,speed_mph=10.,deadline_mono=10.02))
        link.ser.write.assert_called_once_with(b'0.00 10.00 0.0000 0\n')
        link.ser.write.reset_mock()
        with self.assertRaises(ValueError):link.command(0.,speed_mph=10.01)
        link.ser.write.assert_not_called()

    def test_no_write_after_deadline(self):
        link=self.link()
        with patch('dbw_serial.time.monotonic',return_value=10.):
            self.assertFalse(link.command(0.,speed_mph=1.,deadline_mono=9.))
        link.ser.write.assert_not_called()

    def test_write_failure_never_retries_or_reconnects(self):
        link=self.link();port=link.ser;port.write.side_effect=OSError('USB lost')
        with patch('dbw_serial.time.monotonic',return_value=10.):
            self.assertFalse(link.command(0.,speed_mph=1.,deadline_mono=10.02))
            self.assertFalse(link.command(0.,speed_mph=1.,deadline_mono=10.02))
        self.assertEqual(port.write.call_count,1);self.assertTrue(link.failed)

    def test_partial_write_latches_fault(self):
        link=self.link();link.ser.write.side_effect=lambda line:len(line)-1
        with patch('dbw_serial.time.monotonic',return_value=10.):
            self.assertFalse(link.command(0.,deadline_mono=10.02))
        self.assertTrue(link.failed)

    def test_close_never_sends_release_or_centering_commands(self):
        link=self.link();port=link.ser;link.moving=True;link.close()
        port.write.assert_not_called();port.close.assert_called_once()

    def test_bad_numeric_values_do_not_reach_serial(self):
        for kwargs in ({'deg':float('nan')},{'deg':True},{'deg':641.},
                       {'deg':0.,'speed_mph':-1.},{'deg':0.,'pedal':2.}):
            link=self.link()
            with self.assertRaises(ValueError):link.command(**kwargs)
            link.ser.write.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def rejects(self,args):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(args)

    def test_default_is_dry_run(self):
        self.assertTrue(parse_args([]).dry_run)

    def test_default_speed_and_cli_boundary(self):
        self.assertEqual(parse_args([]).max_speed_mph,6.)
        self.assertEqual(parse_args(['--max-speed-mph','10']).max_speed_mph,10.)
        self.assertEqual(parse_args(['--max-speed-mph','1']).max_speed_mph,1.)
        self.rejects(['--max-speed-mph','10.01'])

    def test_hardware_requires_real_configuration(self):
        self.rejects(['--actuate'])
        template=Path(__file__).resolve().parents[1]/'hardware.example.json'
        self.rejects(['--actuate','--hardware-config',str(template)])

    def test_invalid_cli_limits_rejected(self):
        for option,value in (('--rate','0'),('--rate','nan'),('--cmd-timeout','-1'),
                             ('--max-speed-mph','inf'),('--max-pedal','2'),('--kp','-1')):
            self.rejects([option,value])

    def test_hardware_configuration_validation_does_not_open_adapter(self):
        template=Path(__file__).resolve().parents[1]/'hardware.example.json'
        hardware=json.loads(template.read_text())
        for key in list(hardware):
            if key.endswith('_verified'):hardware[key]=True
        hardware.update(baud=115200,max_steering_wheel_deg=30.,
                        max_steering_wheel_rate_deg_s=20.,firmware_heartbeat_timeout_s=.2,
                        verification_notes='Synthetic test fixture; not actual hardware verification.')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.json'
            path.write_text(json.dumps(hardware))
            with patch('bev_dbw_bridge.DBW') as constructor:
                args=parse_args(['--actuate','--hardware-config',str(path)])
                self.assertFalse(args.dry_run);constructor.assert_not_called()
            self.rejects(['--actuate','--hardware-config',str(path),'--lon-mode','pedal'])
            for key in [key for key in hardware if key.endswith('_verified')]:
                with self.subTest(key=key):
                    path.write_text(json.dumps({**hardware,key:False}))
                    self.rejects(['--actuate','--hardware-config',str(path)])

    def test_operator_managed_external_mode_keeps_unknown_checks_explicit(self):
        path=Path(__file__).resolve().parents[1]/'hardware.external.json'
        with patch('bev_dbw_bridge.DBW') as constructor:
            args=parse_args(['--actuate','--hardware-config',str(path),'--max-speed-mph','10'])
            self.assertEqual(args.max_speed_mph,10.)
            self.assertEqual(args.hardware['safety_mode'],'operator_managed_external')
            self.assertIs(args.hardware['protocol_verified'],False)
            self.assertIsNone(args.hardware['firmware_heartbeat_timeout_s'])
            constructor.assert_not_called()
        self.rejects(['--actuate','--hardware-config',str(path),'--max-speed-mph','10.01'])

    def test_no_creep_without_feedback(self):
        controller=PedalPI()
        with self.assertRaises(ValueError):controller.update(1.,None,.02)

    def test_pi_stop_and_saturation(self):
        controller=PedalPI()
        for _ in range(100):self.assertLessEqual(controller.update(10.,0.,.02),.5)
        self.assertEqual(controller.integral,0.)
        self.assertEqual(controller.update(0.,1.,.02),-.2)


class NodeFailureTests(unittest.TestCase):
    def node(self):
        node=object.__new__(BevDbwBridge)
        node.a=Mock(dry_run=False,cmd_timeout=.3,source_timeout=.5,engage_timeout=.3)
        node.dbw=Mock()
        node.transport_fault=False;node.stop_latched=False;node.sent_any=False
        node.gate=InputGate();node.pi=PedalPI()
        node.event=Mock();node.get_logger=Mock(return_value=Mock())
        node.stop_reason=None;node.last_target_mono=None;node.last_engage_mono=None;node.last_output=None
        node.get_clock=Mock(return_value=Mock(now=Mock(return_value=Mock(nanoseconds=10_000_000_000))))
        return node

    def test_stop_latch_records_cause_once(self):
        node=self.node();node.a=parse_args(['--dry-run']);node.sent_any=True
        self.assertTrue(node.gate.receive(message(),10.,10_000_000_000))
        node.last_target_mono=10.;node.gate.engage(False,10.);node.gate.engage(True,10.);node.last_engage_mono=10.
        with patch('bev_dbw_bridge.time.monotonic',return_value=10.4):
            self.assertFalse(node.gate.ready(10.4,10_400_000_000))
            node.latch_stop('not_ready')
            node.latch_stop('target_rejected')
        self.assertEqual(node.stop_reason['cause'],'not_ready')
        self.assertEqual(node.stop_reason['gate_reason'],'target_expired_release_required')
        self.assertAlmostEqual(node.stop_reason['target_receipt_age_s'],.4)
        node.event.assert_called_once()
        self.assertEqual(node.event.call_args.args[0],'stop_latched')

    def test_false_serial_return_latches_and_closes(self):
        node=self.node();port=node.dbw;port.command.return_value=False
        self.assertFalse(node.write(dict(deg=1.,speed_mph=1.,pedal=0.,mode=0),20.))
        self.assertTrue(node.transport_fault);port.close.assert_called_once()
        self.assertFalse(node.write(dict(deg=1.,speed_mph=1.,pedal=0.,mode=0),20.))
        self.assertEqual(port.command.call_count,1)

    def test_stop_intent_holds_steering_and_requests_zero_speed(self):
        node=self.node();node.last_sent_steer=35.;node.write=Mock(return_value=True)
        node.stop_command(10.)
        command=node.write.call_args.args[0]
        self.assertEqual(command,dict(deg=35.,speed_mph=0.,pedal=0.,mode=0))

    def test_inactive_shutdown_has_no_writes(self):
        node=self.node();port=node.dbw;node.log_file=Mock()
        node.shutdown()
        port.command.assert_not_called();port.close.assert_called_once()

    def test_logging_failure_disables_positive_commands(self):
        node=self.node();node.log_file=Mock();node.log_file.write.side_effect=OSError('disk full')
        node.log_failed=False;node.sent_any=True
        BevDbwBridge.event(node,'test')
        self.assertTrue(node.log_failed);self.assertTrue(node.stop_latched)

    def test_speed_ceiling_preserves_lower_and_zero_targets(self):
        for target_mps,limit_mph,expected_mph in ((10.,6.,6.),(2.68224,6.,6.),
                                                  (1.,6.,2.2369363),(10.,1.,1.),(0.,6.,0.)):
            with self.subTest(target_mps=target_mps,limit_mph=limit_mph):
                node=self.node()
                node.a=parse_args(['--dry-run','--max-speed-mph',str(limit_mph)])
                node.last_tick=None;node.last_status=10.;node.last_dry_log=10.
                node.last_sent_steer=0.;node.log_failed=False
                node.hardware=dict(steer_sign=-1,steer_ratio=17.,max_steering_wheel_deg=90.,
                                   max_steering_wheel_rate_deg_s=90.)
                node.get_parameter=Mock(return_value=Mock(value=False))
                node.get_clock=Mock(return_value=Mock(now=Mock(return_value=Mock(nanoseconds=10_000_000_000))))
                self.assertTrue(node.gate.receive(message(target_speed_mps=target_mps),10.,10_000_000_000))
                node.gate.engage(False,10.);node.gate.engage(True,10.)
                node.write=Mock(return_value=True)
                with patch('bev_dbw_bridge.time.monotonic',return_value=10.):node.tick()
                self.assertAlmostEqual(node.write.call_args.args[0]['speed_mph'],expected_mph)


if __name__=='__main__':unittest.main()
