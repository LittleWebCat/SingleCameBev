"""Test SBF framing, fix validity, epoch alignment and read-only reconnects."""
from binascii import crc_hqx
import importlib.util
import math
from pathlib import Path
import queue
import struct
import sys
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("septentrio_gps", ROOT / "tools/septentrio_gps.py")
GPS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GPS
SPEC.loader.exec_module(GPS)


def finish(block_id, payload, revision=0):
    block = bytearray(b"$@\0\0" + struct.pack("<HH", block_id | revision << 13, 8 + len(payload)) + payload)
    struct.pack_into("<H", block, 2, crc_hqx(block[4:], 0))
    return bytes(block)


def pvt_block(mode=4, error=0, latitude=.5, longitude=-1.0, height=150.0,
              week=2440, tow_ms=123400, nr_sv=14, revision=2):
    payload = bytearray(88 if revision == 2 else 80)
    struct.pack_into("<IHBBddd", payload, 0, tow_ms, week, mode, error, latitude, longitude, height)
    payload[64:67] = bytes((0, 0, nr_sv))
    struct.pack_into("<I", payload, 72, (1 << 0) | (1 << 8) | (1 << 13) | (1 << 17))
    return finish(4007, payload, revision)


def cov_block(week=2440, tow_ms=123400, mode=4, error=0,
              values=(4., 9., 16., 1., .5, .6, 0., .7, 0., 0.)):
    return finish(5906, struct.pack("<IHBB10f", tow_ms, week, mode, error, *values))


def envelope(block, receipt_s=10.0, connection_id=1):
    return GPS.ReceivedBlock(block, int(receipt_s * 1e9), receipt_s, connection_id)


class SbfFramingTests(unittest.TestCase):
    def test_partial_chunks_preserve_exact_blocks_and_skip_ascii(self):
        first, second = pvt_block(), cov_block()
        parser = GPS.SbfStreamParser()
        results = []
        stream = b"receiver status\r\n" + first + second
        for start in range(0, len(stream), 7):
            results.extend(parser.feed(stream[start:start + 7]))
        self.assertEqual(results, [first, second])
        self.assertFalse(parser.buffer)
        self.assertEqual(parser.valid_blocks, 2)

    def test_bad_crc_and_bad_length_resynchronize_to_next_valid_packet(self):
        damaged = bytearray(pvt_block())
        damaged[32] ^= 8
        invalid_length = b"$@\0\0" + struct.pack("<HH", 4007, 7)
        good = pvt_block(tow_ms=123500)
        parser = GPS.SbfStreamParser()
        self.assertEqual(parser.feed(bytes(damaged) + invalid_length + good), [good])
        self.assertEqual(parser.crc_errors, 1)
        self.assertEqual(parser.length_errors, 1)

    def test_standalone_decoder_rejects_crc_layout_and_unrecognized_revision(self):
        damaged = bytearray(pvt_block())
        damaged[24] ^= 1
        for block in (bytes(damaged), finish(4007, bytes(76)), pvt_block(revision=3)):
            with self.subTest(length=len(block)):
                with self.assertRaises(ValueError):
                    GPS.decode_pvt(block)

    def test_unknown_crc_valid_blocks_are_preserved_for_postprocessing(self):
        block = finish(6000, b"\x01\x02\x03\x04")
        self.assertEqual(GPS.SbfStreamParser().feed(block), [block])
        self.assertEqual(GPS.block_header(block), (6000, 0))


class FixAndCovarianceTests(unittest.TestCase):
    def test_no_fix_never_exposes_do_not_use_numbers_or_claims_satellites(self):
        pvt = GPS.decode_pvt(pvt_block(mode=0, error=1, latitude=GPS.DNU,
                                      longitude=GPS.DNU, height=GPS.DNU, nr_sv=255))
        self.assertFalse(pvt["valid_fix"])
        self.assertEqual(pvt["navsat_status"], -1)
        self.assertEqual(pvt["rtk_status"], "no_fix")
        self.assertIsNone(pvt["satellites_used"])
        self.assertTrue(all(math.isnan(pvt[key]) for key in ("latitude", "longitude", "altitude")))

    def test_valid_rtk_retains_class_but_uses_standard_gbas_status(self):
        for mode, rtk in ((4, "fixed"), (5, "float"), (7, "fixed"), (8, "float")):
            pvt = GPS.decode_pvt(pvt_block(mode=mode))
            self.assertTrue(pvt["valid_fix"])
            self.assertEqual(pvt["rtk_status"], rtk)
            self.assertEqual(pvt["navsat_status"], 2)
            self.assertAlmostEqual(pvt["latitude"], math.degrees(.5))
            self.assertEqual(pvt["service"], 15)

    def test_errors_unknown_modes_nonfinite_dnu_and_out_of_range_are_invalid(self):
        bad_values = ({"error": 2}, {"mode": 9}, {"latitude": math.inf},
                      {"height": GPS.DNU}, {"longitude": math.nan},
                      {"latitude": 2.0}, {"longitude": 4.0})
        for fields in bad_values:
            with self.subTest(fields=fields):
                self.assertFalse(GPS.decode_pvt(pvt_block(**fields))["valid_fix"])

    def test_covariance_is_square_metres_and_reorders_north_east_up(self):
        covariance = GPS.decode_covariance(cov_block())
        self.assertTrue(covariance["valid"])
        for actual, expected in zip(covariance["enu"], (9., .5, .7, .5, 4., .6, .7, .6, 16.)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_bad_covariance_never_becomes_known_accuracy(self):
        for fields in ({"error": 1}, {"mode": 0},
                       {"values": (GPS.DNU, 9., 16., 1., .5, .6, 0., .7, 0., 0.)},
                       {"values": (4., 9., 16., 1., 50., .6, 0., .7, 0., 0.)}):
            with self.subTest(fields=fields):
                cov = GPS.decode_covariance(cov_block(**fields))
                self.assertFalse(cov["valid"])
                self.assertIsNone(cov["enu"])

    def test_matching_epoch_covariance_works_in_both_arrival_orders(self):
        for covariance_first in (False, True):
            assembler = GPS.EpochAssembler()
            pvt, cov = pvt_block(), cov_block()
            actions = ((assembler.add_pvt, GPS.decode_pvt(pvt), envelope(pvt)),
                       (assembler.add_covariance, GPS.decode_covariance(cov), envelope(cov)))
            for add, decoded, received in (reversed(actions) if covariance_first else actions):
                add(decoded, received)
            results = assembler.ready(10.01)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0][0]["covariance_known"])
            self.assertEqual(results[0][1].receipt_ns, 10000000000)
            self.assertFalse(assembler.ready(10.1))

    def test_mismatched_week_tow_or_mode_remains_unknown_after_wait(self):
        for fields in ({"week": 2439}, {"tow_ms": 123300}, {"mode": 5}):
            assembler = GPS.EpochAssembler()
            pvt, cov = pvt_block(), cov_block(**fields)
            assembler.add_pvt(GPS.decode_pvt(pvt), envelope(pvt))
            assembler.add_covariance(GPS.decode_covariance(cov), envelope(cov))
            self.assertFalse(assembler.ready(10.01))
            results = assembler.ready(10.06)
            self.assertFalse(results[0][0]["covariance_known"])
            self.assertTrue(all(math.isnan(value) for value in results[0][0]["position_covariance"]))

    def test_reordered_pvts_publish_in_epoch_order_and_old_epochs_do_not_repeat(self):
        assembler = GPS.EpochAssembler()
        for tow_ms in (123500, 123400):
            block = pvt_block(tow_ms=tow_ms)
            assembler.add_pvt(GPS.decode_pvt(block), envelope(block))
        self.assertEqual([pvt["tow_ms"] for pvt, _ in assembler.ready(10.1)], [123400, 123500])
        block = pvt_block(tow_ms=123400)
        assembler.add_pvt(GPS.decode_pvt(block), envelope(block, 10.2))
        self.assertFalse(assembler.ready(10.3))
        self.assertEqual(assembler.ignored_old_pvt, 1)

    def test_stale_queued_positions_are_not_published_and_reconnect_clears_covariance(self):
        assembler = GPS.EpochAssembler()
        block = pvt_block()
        assembler.add_pvt(GPS.decode_pvt(block), envelope(block))
        self.assertFalse(assembler.ready(11.1))
        self.assertEqual(assembler.dropped_stale_pvt, 1)
        cov = cov_block()
        assembler.add_covariance(GPS.decode_covariance(cov), envelope(cov, 11.2, 1))
        assembler.add_pvt(GPS.decode_pvt(block), envelope(block, 11.3, 2))
        result = assembler.ready(11.4)[0][0]
        self.assertFalse(result["covariance_known"])


class ReadOnlyTcpTests(unittest.TestCase):
    def test_reconnect_discards_incomplete_previous_connection_and_never_sends(self):
        block = pvt_block()
        successful_receive = threading.Event()
        calls = []

        class ReceiveOnlySocket:
            def __init__(self, chunks):
                self.chunks = list(chunks)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def settimeout(self, value):
                self.timeout = value

            def recv(self, size):
                if self.chunks:
                    data = self.chunks.pop(0)
                    if data == block:
                        successful_receive.set()
                    return data
                return b""

            def shutdown(self, how):
                pass

            def send(self, *args):
                raise AssertionError("receiver writes are forbidden")

            sendall = send

        def connect(endpoint, timeout):
            calls.append((endpoint, timeout))
            if len(calls) == 1:
                raise ConnectionRefusedError("fixture receiver unavailable")
            if len(calls) == 2:
                return ReceiveOnlySocket([block[:17]])
            return ReceiveOnlySocket([block])

        config = GPS.validate_config(dict(GPS.DEFAULTS, reconnect_initial_s=.005, reconnect_max_s=.01))
        reader = GPS.TcpReader(config, stamp_provider=lambda: 123456789, connector=connect)
        reader.start()
        try:
            self.assertTrue(successful_receive.wait(2.0))
            received = reader.blocks.get(timeout=2.0)
            self.assertEqual(received.data, block)
            self.assertEqual(received.receipt_ns, 123456789)
            self.assertEqual(received.connection_id, 2)
        finally:
            reader.stop()
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(reader.snapshot()["connection_failures"], 1)
        self.assertGreaterEqual(reader.snapshot()["discarded_bytes"], 17)

    def test_bounded_queue_counts_loss_and_keeps_current_packet(self):
        config = dict(GPS.DEFAULTS, queue_capacity=1)
        reader = GPS.TcpReader(config)
        first, second = envelope(pvt_block()), envelope(pvt_block(tow_ms=123500))
        reader._put(first)
        reader._put(second)
        self.assertEqual(reader.blocks.get_nowait(), second)
        self.assertEqual(reader.snapshot()["queue_drops"], 1)
        with self.assertRaises(queue.Empty):
            reader.blocks.get_nowait()

    def test_invalid_endpoint_and_unbounded_or_invalid_timeouts_are_rejected(self):
        for fields in ({"receiver_port": 0}, {"receiver_port": True},
                       {"receive_timeout_s": math.inf}, {"connect_timeout_s": 1e100}, {"receiver_host": ""},
                       {"reconnect_max_s": .1}, {"queue_capacity": 0},
                       {"covariance_wait_s": 2.0}):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    GPS.validate_config(dict(GPS.DEFAULTS, **fields))

    def test_health_becomes_stale_without_repeating_position_messages(self):
        reader = GPS.TcpReader(GPS.DEFAULTS)
        stats = reader.snapshot()
        stats.update(connected=True, connected_since_s=9., last_block_s=10.)
        pvt = dict(GPS.decode_pvt(pvt_block()), covariance_known=True)
        current = GPS.health_status(stats, pvt, 10., 10.1, 1., {}, {})
        stale = GPS.health_status(stats, pvt, 10., 11.1, 1., {}, {})
        self.assertTrue(current["valid_fix"])
        self.assertFalse(stale["valid_fix"])
        self.assertTrue(stale["stale"])
        self.assertFalse(stale["covariance_known"])
        self.assertEqual(stale["timestamp_source"], "host_ros_receipt")


if __name__ == "__main__":
    unittest.main()
