#!/usr/bin/python3
"""Read an existing Septentrio SBF TCP stream and publish collection topics.

This client never sends receiver commands, corrections or configuration. Header
stamps are the host ROS clock at TCP receipt, not synchronized GNSS timestamps.
The raw topic contains each complete CRC-validated SBF block, byte for byte.

Field contracts: Septentrio's SBF reference (PVTGeodetic 4007 and
PosCovGeodetic 5906), and the official driver's sbf_blocks.hpp and
message_handler.cpp::assembleNavSatFix. PosCovGeodetic uses m^2 for latitude,
longitude and height; its north/east/up entries are reordered to ROS ENU.
https://github.com/septentrio-gnss/septentrio_gnss_driver
https://www.septentrio.com/system/files/support/asterx_sb3_pro_firmware_v4.10.1_reference_guide.pdf
"""

import argparse
from array import array
from binascii import crc_hqx
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import socket
import struct
import sys
import threading
import time


NODE_NAME = "septentrio_gps_bridge"
TOPICS = {
    "fix": "/sensing/gnss/fix",
    "status": "/sensing/gnss/status",
    "raw_sbf": "/sensing/gnss/raw_sbf",
    "diagnostics": "/sensing/gnss/diagnostics",
}
DNU = -2e10
MODE_NAMES = {
    0: "no_pvt", 1: "standalone", 2: "differential", 3: "fixed_location",
    4: "rtk_fixed", 5: "rtk_float", 6: "sbas", 7: "moving_base_rtk_fixed",
    8: "moving_base_rtk_float", 10: "ppp",
}
ERROR_NAMES = {
    0: "none", 1: "not_enough_measurements", 2: "not_enough_ephemerides",
    3: "dop_too_large", 4: "residuals_too_large", 5: "no_convergence",
    6: "not_enough_measurements_after_rejection", 7: "output_prohibited",
    8: "not_enough_corrections", 9: "base_coordinates_unavailable",
    10: "ambiguities_not_fixed",
}
DEFAULTS = {
    "receiver_host": "192.168.3.1", "receiver_port": 28784,
    "receiver_name": "Septentrio mosaic-X5", "frame_id": "gnss_antenna",
    "connect_timeout_s": 2.0, "receive_timeout_s": 1.0,
    "reconnect_initial_s": 0.5, "reconnect_max_s": 5.0,
    "stale_after_s": 1.0, "covariance_wait_s": 0.05,
    "queue_capacity": 1024,
}


def load_config(path):
    """Read the same node-scoped parameter YAML that ROS launch accepts."""
    import yaml
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("GPS config must be a ROS parameter YAML mapping")
    node = document.get(NODE_NAME, document.get("/" + NODE_NAME))
    if not isinstance(node, dict) or not isinstance(node.get("ros__parameters"), dict):
        raise ValueError("GPS config must contain septentrio_gps_bridge.ros__parameters")
    parameters = node["ros__parameters"]
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown GPS parameters: " + ", ".join(sorted(unknown)))
    return validate_config(dict(DEFAULTS, **parameters))


def validate_config(parameters):
    result = dict(parameters)
    for name in ("receiver_host", "receiver_name", "frame_id"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(name + " must be a nonempty string")
    for name, lower, upper in (("receiver_port", 1, 65535), ("queue_capacity", 1, 65536)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    for name in ("connect_timeout_s", "receive_timeout_s", "reconnect_initial_s",
                 "reconnect_max_s", "stale_after_s", "covariance_wait_s"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 60:
            raise ValueError(name + " must be a finite positive number no greater than 60 seconds")
        result[name] = float(value)
    if result["reconnect_max_s"] < result["reconnect_initial_s"]:
        raise ValueError("reconnect_max_s must be at least reconnect_initial_s")
    if result["covariance_wait_s"] >= result["stale_after_s"]:
        raise ValueError("covariance_wait_s must be less than stale_after_s")
    return result


class SbfStreamParser:
    """Frame fragmented TCP input, reject bad CRCs and resynchronize on $@."""

    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.length_errors = 0
        self.discarded_bytes = 0
        self.valid_blocks = 0

    def feed(self, data):
        self.buffer.extend(data)
        blocks = []
        while True:
            start = self.buffer.find(b"$@")
            if start < 0:
                keep = 1 if self.buffer.endswith(b"$") else 0
                discard = len(self.buffer) - keep
                self.discarded_bytes += discard
                del self.buffer[:discard]
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < 8:
                break
            length = struct.unpack_from("<H", self.buffer, 6)[0]
            if length < 8 or length % 4:
                self.length_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < length:
                break
            block = bytes(self.buffer[:length])
            if crc_hqx(block[4:], 0) != struct.unpack_from("<H", block, 2)[0]:
                self.crc_errors += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            del self.buffer[:length]
            self.valid_blocks += 1
            blocks.append(block)
        return blocks


def block_header(block):
    """Validate a standalone SBF block before decoding any numeric fields."""
    if not isinstance(block, bytes) or len(block) < 8 or block[:2] != b"$@":
        raise ValueError("invalid SBF header")
    crc, raw_id, length = struct.unpack_from("<HHH", block, 2)
    if length != len(block) or length % 4:
        raise ValueError("invalid SBF length")
    if crc_hqx(block[4:], 0) != crc:
        raise ValueError("invalid SBF CRC")
    return raw_id & 8191, raw_id >> 13


def epoch_fields(block):
    if len(block) < 16:
        raise ValueError("SBF block lacks an epoch")
    tow_ms, week = struct.unpack_from("<IH", block, 8)
    return week, tow_ms


def epoch_key(week, tow_ms):
    return (week, tow_ms) if week != 65535 and 0 <= tow_ms < 604800000 else None


def usable_number(value):
    return math.isfinite(value) and value != DNU


def signal_services(signal_info):
    # Follow the official driver's SignalInfo->NavSatStatus service masks.
    return (int(bool(signal_info & 0x3f)) |
            (int(bool(signal_info & sum(1 << bit for bit in range(8, 13)))) << 1) |
            (int(bool(signal_info & sum(1 << bit for bit in (13, 14, 28, 29, 30)))) << 2) |
            (int(bool(signal_info & sum(1 << bit for bit in (17, 19, 20, 21, 22)))) << 3))


def decode_pvt(block):
    block_id, revision = block_header(block)
    if block_id != 4007 or revision not in (0, 1, 2):
        raise ValueError("unsupported PVTGeodetic block/revision")
    if len(block) != (96 if revision == 2 else 88):
        raise ValueError("invalid PVTGeodetic layout")
    week, tow_ms = epoch_fields(block)
    mode, error = struct.unpack_from("<BB", block, 14)
    latitude, longitude, height = struct.unpack_from("<ddd", block, 16)
    solution = mode & 15
    reasons = []
    if solution not in MODE_NAMES or solution == 0:
        reasons.append("no_valid_pvt_mode")
    if error:
        reasons.append("receiver_pvt_error")
    if not all(usable_number(value) for value in (latitude, longitude, height)):
        reasons.append("invalid_or_do_not_use_coordinates")
    elif not (-math.pi / 2 <= latitude <= math.pi / 2 and -math.pi <= longitude <= math.pi):
        reasons.append("coordinates_out_of_range")
    valid = not reasons
    nav_status = -1 if not valid else (1 if solution == 6 else (2 if solution in (2, 4, 5, 7, 8, 10) else 0))
    rtk = "no_fix" if not valid else ("fixed" if solution in (4, 7) else ("float" if solution in (5, 8) else "not_rtk"))
    nr_sv = block[74]
    return {
        "gnss_week": week, "tow_ms": tow_ms, "mode": mode,
        "solution_mode": solution, "mode_name": MODE_NAMES.get(solution, "unknown"),
        "error": error, "error_name": ERROR_NAMES.get(error, "unknown"),
        "rtk_status": rtk, "valid_fix": valid, "invalid_reasons": reasons,
        "navsat_status": nav_status,
        "latitude": math.degrees(latitude) if valid else math.nan,
        "longitude": math.degrees(longitude) if valid else math.nan,
        "altitude": height if valid else math.nan,
        "satellites_used": None if nr_sv == 255 else nr_sv,
        "service": signal_services(struct.unpack_from("<I", block, 80)[0]) if valid else 0,
        "datum": None if block[73] == 255 else block[73],
        "time_system": None if block[72] == 255 else block[72],
        "sbf_revision": revision,
    }


def decode_covariance(block):
    block_id, revision = block_header(block)
    if block_id != 5906 or revision != 0 or len(block) != 56:
        raise ValueError("unsupported PosCovGeodetic layout/revision")
    week, tow_ms = epoch_fields(block)
    mode, error = struct.unpack_from("<BB", block, 14)
    nn, ee, uu, _, ne, nu, _, eu, _, _ = struct.unpack_from("<10f", block, 16)
    entries = (nn, ee, uu, ne, nu, eu)
    valid = (not error and (mode & 15) in MODE_NAMES and (mode & 15) != 0 and
             epoch_key(week, tow_ms) is not None and
             all(usable_number(value) for value in entries) and min(nn, ee, uu) >= 0)
    # A covariance matrix must be positive semidefinite, allowing float roundoff.
    if valid:
        scale = max(nn, ee, uu, 1.0)
        tolerance = 1e-6
        valid = (ne * ne <= nn * ee + tolerance * scale**2 and
                 nu * nu <= nn * uu + tolerance * scale**2 and
                 eu * eu <= ee * uu + tolerance * scale**2 and
                 nn * ee * uu + 2 * ne * nu * eu - nn * eu * eu - ee * nu * nu - uu * ne * ne >= -tolerance * scale**3)
    return {"gnss_week": week, "tow_ms": tow_ms, "mode": mode, "error": error,
            "valid": valid, "enu": [ee, ne, eu, ne, nn, nu, eu, nu, uu] if valid else None}


def decode_receiver_metadata(block):
    block_id, revision = block_header(block)
    week, tow_ms = epoch_fields(block)
    if block_id == 5914:
        if revision != 0 or len(block) != 24:
            raise ValueError("unsupported ReceiverTime layout/revision")
        year, month, day, hour, minute, second, leap_seconds, sync_level = struct.unpack_from("<6BbB", block, 14)
        return "receiver_time", {
            "gnss_week": week, "tow_ms": tow_ms, "utc_year_since_2000": year,
            "utc_month": month, "utc_day": day, "utc_hour": hour,
            "utc_minute": minute, "utc_second": second,
            "delta_leap_seconds": leap_seconds, "sync_level": sync_level,
        }
    if block_id == 4014:
        if revision not in (0, 1) or len(block) < 32:
            raise ValueError("unsupported ReceiverStatus layout/revision")
        count, subblock_length = block[28], block[29]
        if count > 18 or (count and subblock_length < 4) or len(block) != ((32 + count * subblock_length + 3) // 4) * 4:
            raise ValueError("invalid ReceiverStatus subblock layout")
        uptime, status, error = struct.unpack_from("<III", block, 16)
        return "receiver_status", {
            "gnss_week": week, "tow_ms": tow_ms, "cpu_load": block[14],
            "external_error": block[15], "uptime_s": uptime,
            "rx_status": status, "rx_error": error, "command_count": block[30],
            "temperature_raw": block[31],
        }
    return None, None


@dataclass(frozen=True)
class ReceivedBlock:
    data: bytes
    receipt_ns: int
    monotonic_s: float
    connection_id: int


class EpochAssembler:
    """Wait briefly for matching covariance in either packet order, once per PVT."""

    def __init__(self, covariance_wait_s=0.05, stale_after_s=1.0, cache_size=64):
        self.wait_s = covariance_wait_s
        self.stale_after_s = stale_after_s
        self.cache_size = cache_size
        self.pending = OrderedDict()
        self.covariances = OrderedDict()
        self.last_published_epoch = None
        self.sequence = 0
        self.dropped_stale_pvt = 0
        self.ignored_old_pvt = 0
        self.connection_id = None

    def reset_connection(self, connection_id):
        if self.connection_id != connection_id:
            self.pending.clear()
            self.covariances.clear()
            self.last_published_epoch = None
            self.connection_id = connection_id

    def add_pvt(self, pvt, envelope):
        self.reset_connection(envelope.connection_id)
        key = epoch_key(pvt["gnss_week"], pvt["tow_ms"])
        if key is not None and self.last_published_epoch is not None and key <= self.last_published_epoch:
            self.ignored_old_pvt += 1
            return
        self.sequence += 1
        pending_key = key if key is not None else ("invalid_time", self.sequence)
        if pending_key in self.pending:
            self.ignored_old_pvt += 1
            return
        self.pending[pending_key] = (pvt, envelope)
        if len(self.pending) > self.cache_size:
            self.pending.popitem(last=False)
            self.dropped_stale_pvt += 1

    def add_covariance(self, covariance, envelope):
        self.reset_connection(envelope.connection_id)
        key = epoch_key(covariance["gnss_week"], covariance["tow_ms"])
        if key is not None:
            self.covariances[key] = covariance
            self.covariances.move_to_end(key)
            if len(self.covariances) > self.cache_size:
                self.covariances.popitem(last=False)

    def ready(self, now_s):
        ready = []
        # Publish increasing receiver epochs even if several PVT packets reorder.
        ordered = sorted(self.pending.items(), key=lambda item: (
            epoch_key(item[1][0]["gnss_week"], item[1][0]["tow_ms"]) or (65536, item[1][1].receipt_ns)))
        for pending_key, (pvt, envelope) in ordered:
            age = max(0.0, now_s - envelope.monotonic_s)
            key = epoch_key(pvt["gnss_week"], pvt["tow_ms"])
            cov = self.covariances.get(key)
            matching = (pvt["valid_fix"] and cov is not None and cov["valid"] and
                        (cov["mode"] & 15) == pvt["solution_mode"])
            if age >= self.stale_after_s:
                del self.pending[pending_key]
                self.dropped_stale_pvt += 1
                continue
            if not matching and age < self.wait_s:
                break
            del self.pending[pending_key]
            if key is not None and self.last_published_epoch is not None and key <= self.last_published_epoch:
                self.ignored_old_pvt += 1
                continue
            result = dict(pvt, covariance_known=bool(matching),
                          position_covariance=list(cov["enu"]) if matching else [math.nan] * 9)
            ready.append((result, envelope))
            if key is not None:
                self.last_published_epoch = key
        return ready


class TcpReader(threading.Thread):
    """Only connect and receive; all ROS message construction runs in the timer."""

    def __init__(self, config, stamp_provider=time.time_ns, connector=socket.create_connection):
        super().__init__(name="septentrio_sbf_tcp", daemon=True)
        self.config = config
        self.stamp_provider = stamp_provider
        self.connector = connector
        self.blocks = queue.Queue(maxsize=config["queue_capacity"])
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_socket = None
        self.stats = {"connected": False, "connected_since_s": None,
                      "last_block_s": None, "connection_id": 0,
                      "connection_attempts": 0, "connection_failures": 0,
                      "disconnects": 0, "queue_drops": 0, "crc_errors": 0,
                      "length_errors": 0, "discarded_bytes": 0,
                      "validated_blocks": 0, "last_error": None}

    def snapshot(self):
        with self.lock:
            return dict(self.stats)

    def _put(self, envelope):
        try:
            self.blocks.put_nowait(envelope)
        except queue.Full:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                pass
            with self.lock:
                self.stats["queue_drops"] += 1
            try:
                self.blocks.put_nowait(envelope)
            except queue.Full:
                with self.lock:
                    self.stats["queue_drops"] += 1

    def stop(self):
        self.stop_event.set()
        with self.lock:
            active_socket = self.active_socket
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.join(timeout=self.config["connect_timeout_s"] + self.config["receive_timeout_s"] + 0.5)

    def run(self):
        delay = self.config["reconnect_initial_s"]
        while not self.stop_event.is_set():
            with self.lock:
                self.stats["connection_attempts"] += 1
            parser = SbfStreamParser()
            parser_counts = {"crc_errors": 0, "length_errors": 0,
                             "discarded_bytes": 0, "valid_blocks": 0}
            connection_id = None
            had_blocks = False
            try:
                with self.connector((self.config["receiver_host"], self.config["receiver_port"]),
                                    timeout=self.config["connect_timeout_s"]) as connection:
                    connection.settimeout(self.config["receive_timeout_s"])
                    with self.lock:
                        self.active_socket = connection
                        self.stats["connection_id"] += 1
                        connection_id = self.stats["connection_id"]
                        self.stats.update(connected=True, connected_since_s=time.monotonic(), last_error=None)
                    while not self.stop_event.is_set():
                        # A timeout closes an unresponsive stream and reconnects.
                        data = connection.recv(65536)
                        if not data:
                            raise ConnectionError("receiver closed the TCP stream")
                        receipt_ns, receipt_s = self.stamp_provider(), time.monotonic()
                        blocks = parser.feed(data)
                        with self.lock:
                            for field, old_count in parser_counts.items():
                                count = getattr(parser, field)
                                target = "validated_blocks" if field == "valid_blocks" else field
                                self.stats[target] += count - old_count
                                parser_counts[field] = count
                            if blocks:
                                had_blocks = True
                                self.stats["last_block_s"] = receipt_s
                        for block in blocks:
                            self._put(ReceivedBlock(block, receipt_ns, receipt_s, connection_id))
            except (OSError, ConnectionError) as error:
                with self.lock:
                    if connection_id is None:
                        self.stats["connection_failures"] += 1
                    self.stats["last_error"] = type(error).__name__ + ": " + str(error)
            finally:
                with self.lock:
                    if connection_id is not None:
                        self.stats["disconnects"] += 1
                    self.active_socket = None
                    self.stats.update(connected=False, connected_since_s=None)
                    self.stats["discarded_bytes"] += len(parser.buffer)
            if had_blocks:
                delay = self.config["reconnect_initial_s"]
            if self.stop_event.wait(delay):
                break
            if not had_blocks:
                delay = min(delay * 2, self.config["reconnect_max_s"])


def health_status(reader_stats, last_pvt, last_pvt_s, now_s, stale_after_s, counters, metadata):
    block_s = reader_stats["last_block_s"]
    block_age = None if block_s is None else max(0.0, now_s - block_s)
    pvt_age = None if last_pvt_s is None else max(0.0, now_s - last_pvt_s)
    connected_s = reader_stats["connected_since_s"]
    result = {name: None for name in ("mode", "solution_mode", "mode_name", "error", "error_name",
                                     "rtk_status", "gnss_week", "tow_ms", "satellites_used", "datum", "time_system")}
    if last_pvt is not None:
        for name in result:
            result[name] = last_pvt[name]
    stale = pvt_age is None or pvt_age >= stale_after_s
    result.update({
        "connected": reader_stats["connected"], "stale": stale,
        "valid_fix": bool(last_pvt and last_pvt["valid_fix"] and not stale and reader_stats["connected"]),
        "covariance_known": bool(last_pvt and last_pvt.get("covariance_known", False) and not stale),
        "receiver_connection_age_s": None if connected_s is None else max(0.0, now_s - connected_s),
        "last_block_age_s": block_age, "last_pvt_age_s": pvt_age,
        "timestamp_source": "host_ros_receipt", "gnss_clock_synchronized": False,
        "counters": dict(counters, **{name: reader_stats[name] for name in (
            "connection_attempts", "connection_failures", "disconnects", "queue_drops", "crc_errors",
            "length_errors", "discarded_bytes", "validated_blocks")}),
        "last_connection_error": reader_stats["last_error"],
    })
    result.update(metadata)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/sensors/septentrio_gps.yaml")
    args, ros_args = parser.parse_known_args(argv)
    if ros_args and ros_args[0] != "--ros-args":
        parser.error("unrecognized arguments: " + " ".join(ros_args))
    try:
        defaults = load_config(args.config)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from sensor_msgs.msg import NavSatFix
    from std_msgs.msg import String, UInt8MultiArray

    class GpsBridge(Node):
        def __init__(self):
            super().__init__(NODE_NAME)
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            self.config = validate_config({name: self.get_parameter(name).value for name in defaults})
            self.publishers_by_name = {name: self.create_publisher(message_type, TOPICS[name], depth)
                                      for name, message_type, depth in (
                                          ("fix", NavSatFix, 100), ("status", String, 100),
                                          ("raw_sbf", UInt8MultiArray, 1000), ("diagnostics", DiagnosticArray, 10))}
            self.assembler = EpochAssembler(self.config["covariance_wait_s"], self.config["stale_after_s"])
            self.reader = TcpReader(self.config, stamp_provider=lambda: self.get_clock().now().nanoseconds)
            self.last_pvt, self.last_pvt_s = None, None
            self.metadata = {"receiver_name": self.config["receiver_name"],
                             "receiver_host": self.config["receiver_host"], "receiver_port": self.config["receiver_port"],
                             "receiver_status": None, "receiver_time": None}
            self.counters = {"decode_errors": 0, "published_fixes": 0}
            self.last_health_s = 0.0
            self.timer = self.create_timer(0.02, self.poll)
            self.reader.start()
            self.get_logger().info(f"Receiving existing SBF stream at {self.config['receiver_host']}:{self.config['receiver_port']}; host receipt stamps")

        def poll(self):
            for _ in range(256):
                try:
                    envelope = self.reader.blocks.get_nowait()
                except queue.Empty:
                    break
                raw = UInt8MultiArray()
                raw.data = array("B", envelope.data)
                self.publishers_by_name["raw_sbf"].publish(raw)
                try:
                    block_id, _ = block_header(envelope.data)
                    if block_id == 4007:
                        self.assembler.add_pvt(decode_pvt(envelope.data), envelope)
                    elif block_id == 5906:
                        self.assembler.add_covariance(decode_covariance(envelope.data), envelope)
                    elif block_id in (4014, 5914):
                        name, value = decode_receiver_metadata(envelope.data)
                        self.metadata[name] = value
                except ValueError:
                    self.counters["decode_errors"] += 1
            now_s = time.monotonic()
            for pvt, envelope in self.assembler.ready(now_s):
                fix = NavSatFix()
                fix.header.stamp.sec, fix.header.stamp.nanosec = divmod(envelope.receipt_ns, 1000000000)
                fix.header.frame_id = self.config["frame_id"]
                fix.status.status, fix.status.service = pvt["navsat_status"], pvt["service"]
                fix.latitude, fix.longitude, fix.altitude = pvt["latitude"], pvt["longitude"], pvt["altitude"]
                fix.position_covariance = pvt["position_covariance"]
                fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN if pvt["covariance_known"] else NavSatFix.COVARIANCE_TYPE_UNKNOWN
                self.publishers_by_name["fix"].publish(fix)
                self.counters["published_fixes"] += 1
                self.last_pvt, self.last_pvt_s = pvt, envelope.monotonic_s
                self.publish_health(now_s, diagnostics=False)
            if now_s - self.last_health_s >= 0.5:
                self.publish_health(now_s, diagnostics=True)
                self.last_health_s = now_s

        def publish_health(self, now_s, diagnostics):
            counters = dict(self.counters, dropped_stale_pvt=self.assembler.dropped_stale_pvt,
                            ignored_old_pvt=self.assembler.ignored_old_pvt)
            status = health_status(self.reader.snapshot(), self.last_pvt, self.last_pvt_s,
                                   now_s, self.config["stale_after_s"], counters, self.metadata)
            message = String()
            message.data = json.dumps(status, sort_keys=True, allow_nan=False)
            self.publishers_by_name["status"].publish(message)
            if not diagnostics:
                return
            diagnostic = DiagnosticStatus()
            diagnostic.name, diagnostic.hardware_id = "septentrio_gps", self.config["receiver_name"]
            if not status["connected"] or status["stale"]:
                diagnostic.level, diagnostic.message = DiagnosticStatus.ERROR, "Receiver disconnected or PVT stale"
            elif not status["valid_fix"]:
                diagnostic.level, diagnostic.message = DiagnosticStatus.WARN, "Receiver streaming; no usable GNSS fix"
            elif not status["covariance_known"] or counters["decode_errors"] or status["counters"]["queue_drops"]:
                diagnostic.level, diagnostic.message = DiagnosticStatus.WARN, "GNSS fix received; inspect covariance or data counters"
            else:
                diagnostic.level, diagnostic.message = DiagnosticStatus.OK, "Fresh GNSS fix and matching covariance"
            diagnostic.values = [KeyValue(key=key, value=json.dumps(value, sort_keys=True)) for key, value in status.items()]
            report = DiagnosticArray()
            report.header.stamp = self.get_clock().now().to_msg()
            report.status = [diagnostic]
            self.publishers_by_name["diagnostics"].publish(report)

        def destroy_node(self):
            self.reader.stop()
            return super().destroy_node()

    rclpy.init(args=ros_args)
    node = None
    try:
        node = GpsBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
