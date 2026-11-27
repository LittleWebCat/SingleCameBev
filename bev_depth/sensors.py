from __future__ import annotations

import base64
import json
import math
import re
import struct
import threading
import time
from dataclasses import asdict, dataclass, replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
import serial


SensorKind = Literal["gps", "imu"]
RecorderState = Literal["idle", "recording", "error"]
SerialFactory = Callable[..., Any]
NMEA_PATTERN = re.compile(r"\$[A-Z]{5},")


@dataclass(frozen=True)
class RecorderSnapshot:
    state: RecorderState
    port: str | None
    records: int
    bytes_written: int
    output_path: Path | None
    error: str | None


@dataclass(frozen=True)
class RecordingSession:
    session_id: str
    start_wall_time_ns: int
    start_monotonic_ns: int


@dataclass(frozen=True)
class LiveGpsData:
    port: str | None = None
    connected: bool = False
    received: bool = False
    fix: bool | None = None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float | None = None
    satellites: int | None = None
    speed_kph: float | None = None
    utc: str | None = None
    updated_monotonic_ns: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class LiveImuData:
    port: str | None = None
    connected: bool = False
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    acceleration_mps2: tuple[float, float, float] | None = None
    angular_rate_rps: tuple[float, float, float] | None = None
    packets: int = 0
    updated_monotonic_ns: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class LiveSensorSnapshot:
    gps: LiveGpsData
    imu: LiveImuData


def write_matrix_sensor_record(
    handle,
    *,
    frame_id: int,
    raw_matrix_file: str,
    tracked_matrix_file: str | None,
    wall_time_ns: int,
    monotonic_ns: int,
    camera_capture_monotonic_ns: dict[str, int],
    snapshot: LiveSensorSnapshot,
) -> None:
    """Write the latest decoded GPS/IMU values aligned to one BEV matrix."""

    gps = asdict(snapshot.gps)
    imu = asdict(snapshot.imu)
    gps["age_ns"] = (
        max(0, monotonic_ns - snapshot.gps.updated_monotonic_ns)
        if snapshot.gps.updated_monotonic_ns is not None
        else None
    )
    imu["age_ns"] = (
        max(0, monotonic_ns - snapshot.imu.updated_monotonic_ns)
        if snapshot.imu.updated_monotonic_ns is not None
        else None
    )
    record = {
        "frame_id": frame_id,
        "raw_matrix_file": raw_matrix_file,
        "tracked_matrix_file": tracked_matrix_file,
        "wall_time_ns": wall_time_ns,
        "monotonic_ns": monotonic_ns,
        "camera_capture_monotonic_ns": camera_capture_monotonic_ns,
        "gps": gps,
        "imu": imu,
    }
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def create_recording_session() -> RecordingSession:
    wall_time_ns = time.time_ns()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id += f"_{wall_time_ns % 1_000_000_000:09d}"
    return RecordingSession(
        session_id=session_id,
        start_wall_time_ns=wall_time_ns,
        start_monotonic_ns=time.monotonic_ns(),
    )


def resolve_sensor_port(kind: SensorKind, configured_port: str = "auto") -> str | None:
    if configured_port.lower() != "auto":
        return configured_port

    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        tokens = ("septentrio",) if kind == "gps" else ("xsens", "mti")
        for candidate in sorted(by_id.iterdir()):
            name = candidate.name.lower()
            if any(token in name for token in tokens):
                if kind == "gps" and "if02" not in name:
                    continue
                return str(candidate)

    fallbacks = ("/dev/ttyACM0",) if kind == "gps" else ("/dev/ttyUSB0",)
    return next((path for path in fallbacks if Path(path).exists()), None)


def _is_septentrio_port(port: str) -> bool:
    if "septentrio" in port.lower():
        return True
    by_id = Path("/dev/serial/by-id")
    if not by_id.is_dir():
        return False
    try:
        target = Path(port).resolve()
        return any(
            "septentrio" in candidate.name.lower()
            and candidate.resolve() == target
            for candidate in by_id.iterdir()
        )
    except OSError:
        return False


def _nmea_coordinate(value: str, hemisphere: str) -> float | None:
    if not value or hemisphere not in {"N", "S", "E", "W"}:
        return None
    degree_digits = 2 if hemisphere in {"N", "S"} else 3
    degrees = float(value[:degree_digits])
    minutes = float(value[degree_digits:])
    coordinate = degrees + minutes / 60.0
    return -coordinate if hemisphere in {"S", "W"} else coordinate


def _extract_xbus_frames(buffer: bytearray) -> list[tuple[int, bytes]]:
    frames: list[tuple[int, bytes]] = []
    preamble = b"\xfa\xff"
    while True:
        offset = buffer.find(preamble)
        if offset < 0:
            if buffer[-1:] != preamble[:1]:
                buffer.clear()
            elif len(buffer) > 1:
                del buffer[:-1]
            break
        if offset:
            del buffer[:offset]
        if len(buffer) < 5:
            break
        payload_length = buffer[3]
        if payload_length == 0xFF:
            if len(buffer) < 7:
                break
            payload_length = int.from_bytes(buffer[4:6], "big")
            payload_start = 6
        else:
            payload_start = 4
        frame_length = payload_start + payload_length + 1
        if len(buffer) < frame_length:
            break
        if sum(buffer[1:frame_length]) & 0xFF:
            del buffer[0]
            continue
        frames.append((buffer[2], bytes(buffer[payload_start:-1][:payload_length])))
        del buffer[:frame_length]
    return frames


def _parse_mtdata2(payload: bytes) -> dict[str, Any]:
    values: dict[str, Any] = {}
    offset = 0
    while offset + 3 <= len(payload):
        data_id = int.from_bytes(payload[offset : offset + 2], "big")
        size = payload[offset + 2]
        start = offset + 3
        end = start + size
        if end > len(payload):
            break
        block = payload[start:end]
        base_id = data_id & 0xFFF0
        if size == 12 and base_id in {0x2030, 0x4020, 0x8020}:
            vector = struct.unpack(">fff", block)
            if base_id == 0x2030:
                values["orientation"] = vector
            elif base_id == 0x4020:
                values["acceleration"] = vector
            else:
                values["angular_rate"] = vector
        offset = end
    return values


class LiveSensorMonitor:
    """Continuously decode Septentrio NMEA and Xsens MTData2 for display."""

    def __init__(
        self,
        gps_port: str = "auto",
        imu_port: str = "auto",
        gps_baud: int = 115200,
        imu_baud: int = 115200,
        serial_factory: SerialFactory = serial.Serial,
    ) -> None:
        self.configured_gps_port = gps_port
        self.configured_imu_port = imu_port
        self.gps_baud = gps_baud
        self.imu_baud = imu_baud
        self.serial_factory = serial_factory
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._gps = LiveGpsData()
        self._imu = LiveImuData()

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        ports = {
            "gps": resolve_sensor_port("gps", self.configured_gps_port),
            "imu": resolve_sensor_port("imu", self.configured_imu_port),
        }
        if ports["gps"] is None:
            self._set_gps(error="GPS port not found")
        else:
            self._set_gps(port=ports["gps"])
            self._start_thread("live-gps", self._gps_loop, ports["gps"])
        if ports["imu"] is None:
            self._set_imu(error="IMU port not found")
        else:
            self._set_imu(port=ports["imu"])
            self._start_thread("live-imu", self._imu_loop, ports["imu"])

    def _start_thread(self, name: str, target: Callable[[str], None], port: str) -> None:
        thread = threading.Thread(target=target, args=(port,), name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def snapshot(self) -> LiveSensorSnapshot:
        with self._lock:
            return LiveSensorSnapshot(self._gps, self._imu)

    def _set_gps(self, **changes: Any) -> None:
        with self._lock:
            self._gps = dataclass_replace(self._gps, **changes)

    def _set_imu(self, **changes: Any) -> None:
        with self._lock:
            self._imu = dataclass_replace(self._imu, **changes)

    def _gps_loop(self, port: str) -> None:
        device: Any | None = None
        configured_output = False
        try:
            device = self.serial_factory(
                port=port, baudrate=self.gps_baud, timeout=0.2
            )
            if hasattr(device, "reset_input_buffer"):
                device.reset_input_buffer()
            configured_output = _is_septentrio_port(port)
            if configured_output:
                device.write(b"sno, Stream10, USB1, GGA+RMC, msec100\r\n")
                device.flush()
            self._set_gps(connected=True, error=None)
            while not self._stop_event.is_set():
                raw = bytes(device.readline())
                if raw:
                    self._update_gps(raw)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._set_gps(error=f"{port}: {exc}")
        finally:
            if device is not None:
                if configured_output:
                    try:
                        device.write(b"sno, Stream10, USB1, none, off\r\n")
                        device.flush()
                    except Exception:
                        pass
                try:
                    device.close()
                except Exception:
                    pass
            self._set_gps(connected=False)

    def _update_gps(self, raw: bytes) -> None:
        text = raw.decode("ascii", errors="ignore").strip()
        match = NMEA_PATTERN.search(text)
        if match is None:
            return
        fields = text[match.start() :].split(",")
        sentence = fields[0][-3:]
        changes: dict[str, Any] = {
            "received": True,
            "updated_monotonic_ns": time.monotonic_ns(),
        }
        try:
            if sentence == "GGA" and len(fields) >= 10:
                changes.update(
                    utc=fields[1] or None,
                    latitude_deg=_nmea_coordinate(fields[2], fields[3]),
                    longitude_deg=_nmea_coordinate(fields[4], fields[5]),
                    fix=int(fields[6] or 0) > 0,
                    satellites=int(fields[7] or 0),
                    altitude_m=float(fields[9]) if fields[9] else None,
                )
            elif sentence == "RMC" and len(fields) >= 8:
                changes.update(
                    utc=fields[1] or None,
                    fix=fields[2] == "A",
                    latitude_deg=_nmea_coordinate(fields[3], fields[4]),
                    longitude_deg=_nmea_coordinate(fields[5], fields[6]),
                    speed_kph=float(fields[7] or 0.0) * 1.852,
                )
            else:
                return
        except (ValueError, IndexError):
            return
        self._set_gps(**changes)

    def _imu_loop(self, port: str) -> None:
        device: Any | None = None
        buffer = bytearray()
        try:
            device = self.serial_factory(
                port=port, baudrate=self.imu_baud, timeout=0.2
            )
            if hasattr(device, "reset_input_buffer"):
                device.reset_input_buffer()
            self._set_imu(connected=True, error=None)
            while not self._stop_event.is_set():
                waiting = int(getattr(device, "in_waiting", 0) or 0)
                raw = bytes(device.read(min(max(waiting, 1), 4096)))
                if not raw:
                    continue
                buffer.extend(raw)
                for message_id, payload in _extract_xbus_frames(buffer):
                    if message_id == 0x36:
                        self._update_imu(payload)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._set_imu(error=f"{port}: {exc}")
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
            self._set_imu(connected=False)

    def _update_imu(self, payload: bytes) -> None:
        values = _parse_mtdata2(payload)
        if not values:
            return
        changes: dict[str, Any] = {
            "packets": self.snapshot().imu.packets + 1,
            "updated_monotonic_ns": time.monotonic_ns(),
        }
        orientation = values.get("orientation")
        if orientation is not None:
            changes.update(
                roll_deg=orientation[0],
                pitch_deg=orientation[1],
                yaw_deg=orientation[2],
            )
        if "acceleration" in values:
            changes["acceleration_mps2"] = values["acceleration"]
        if "angular_rate" in values:
            changes["angular_rate_rps"] = values["angular_rate"]
        self._set_imu(**changes)


def _sensor_age(updated_monotonic_ns: int | None) -> str:
    if updated_monotonic_ns is None:
        return "--"
    age_s = max(0.0, (time.monotonic_ns() - updated_monotonic_ns) / 1e9)
    return f"{age_s:.1f}s"


def _draw_fitted_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    max_width: int,
) -> None:
    scale = 0.52
    while scale > 0.3:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= max_width:
            break
        scale -= 0.04
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_live_sensor_panel(
    frame_bgr: np.ndarray, snapshot: LiveSensorSnapshot
) -> np.ndarray:
    """Draw current GPS and IMU values over a display-only frame."""

    width = frame_bgr.shape[1]
    panel_height = min(100, frame_bgr.shape[0])
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.86, frame_bgr, 0.14, 0.0, frame_bgr)
    cv2.putText(
        frame_bgr,
        "LIVE GPS + IMU   [I] hide",
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    gps = snapshot.gps
    if gps.error:
        gps_text = f"GPS ERROR: {gps.error}"
        gps_color = (70, 90, 255)
    elif not gps.connected:
        gps_text = f"GPS: connecting {gps.port or ''}".rstrip()
        gps_color = (0, 190, 255)
    elif not gps.received:
        gps_text = f"GPS: waiting for NMEA on {gps.port}"
        gps_color = (0, 190, 255)
    elif gps.fix:
        location = (
            f"{gps.latitude_deg:.6f}, {gps.longitude_deg:.6f}"
            if gps.latitude_deg is not None and gps.longitude_deg is not None
            else "position pending"
        )
        gps_text = (
            f"GPS FIX: {location} | sats={gps.satellites or 0} "
            f"alt={gps.altitude_m if gps.altitude_m is not None else 0.0:.1f}m "
            f"speed={gps.speed_kph if gps.speed_kph is not None else 0.0:.1f}km/h "
            f"UTC={gps.utc or '--'} age={_sensor_age(gps.updated_monotonic_ns)}"
        )
        gps_color = (80, 240, 100)
    else:
        gps_text = (
            f"GPS NO FIX | sats={gps.satellites or 0} UTC={gps.utc or '--'} "
            f"age={_sensor_age(gps.updated_monotonic_ns)}"
        )
        gps_color = (0, 190, 255)
    _draw_fitted_text(frame_bgr, gps_text, (10, 43), gps_color, width - 20)

    imu = snapshot.imu
    if imu.error:
        imu_text = f"IMU ERROR: {imu.error}"
        motion_text = ""
        imu_color = (70, 90, 255)
    elif not imu.connected:
        imu_text = f"IMU: connecting {imu.port or ''}".rstrip()
        motion_text = ""
        imu_color = (0, 190, 255)
    elif imu.roll_deg is None:
        imu_text = f"IMU: waiting for MTData2 on {imu.port}"
        motion_text = ""
        imu_color = (0, 190, 255)
    else:
        imu_text = (
            f"IMU RPY: {imu.roll_deg:+.1f} {imu.pitch_deg or 0.0:+.1f} "
            f"{imu.yaw_deg or 0.0:+.1f} deg | packets={imu.packets} "
            f"age={_sensor_age(imu.updated_monotonic_ns)}"
        )
        acceleration = imu.acceleration_mps2 or (0.0, 0.0, 0.0)
        angular_rate = tuple(
            math.degrees(value) for value in (imu.angular_rate_rps or (0.0, 0.0, 0.0))
        )
        motion_text = (
            "Accel xyz: "
            f"{acceleration[0]:+.2f} {acceleration[1]:+.2f} {acceleration[2]:+.2f} m/s2 | "
            "Gyro xyz: "
            f"{angular_rate[0]:+.1f} {angular_rate[1]:+.1f} {angular_rate[2]:+.1f} deg/s"
        )
        imu_color = (80, 240, 100)
    _draw_fitted_text(frame_bgr, imu_text, (10, 67), imu_color, width - 20)
    if motion_text and panel_height >= 90:
        _draw_fitted_text(frame_bgr, motion_text, (10, 91), imu_color, width - 20)
    return frame_bgr


class SerialRecorder:
    """Record a serial sensor without blocking camera inference."""

    def __init__(
        self,
        kind: SensorKind,
        output_dir: str | Path,
        port: str = "auto",
        baudrate: int = 115200,
        serial_factory: SerialFactory = serial.Serial,
    ) -> None:
        self.kind = kind
        self.output_dir = Path(output_dir)
        self.configured_port = port
        self.baudrate = baudrate
        self.serial_factory = serial_factory
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any | None = None
        self._state: RecorderState = "idle"
        self._port: str | None = None
        self._records = 0
        self._bytes_written = 0
        self._output_path: Path | None = None
        self._error: str | None = None
        self._shutdown_command: bytes | None = None
        self._session: RecordingSession | None = None

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            return RecorderSnapshot(
                state=self._state,
                port=self._port,
                records=self._records,
                bytes_written=self._bytes_written,
                output_path=self._output_path,
                error=self._error,
            )

    def start(self, session: RecordingSession | None = None) -> bool:
        with self._lock:
            if self._state == "recording":
                return True
            self._error = None
            self._records = 0
            self._bytes_written = 0

        port = resolve_sensor_port(self.kind, self.configured_port)
        if port is None:
            self._set_error(f"{self.kind.upper()} port not found")
            return False

        try:
            device = self.serial_factory(port=port, baudrate=self.baudrate, timeout=0.2)
            if hasattr(device, "reset_input_buffer"):
                device.reset_input_buffer()
            if self.kind == "gps" and "septentrio" in port.lower():
                device.write(b"sno, Stream10, USB1, GGA+RMC, msec100\r\n")
                device.flush()
                self._shutdown_command = b"sno, Stream10, USB1, none, off\r\n"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            session = session or create_recording_session()
            output_path = self.output_dir / f"{self.kind}_{session.session_id}.jsonl"
        except Exception as exc:
            self._set_error(f"{port}: {exc}")
            return False

        with self._lock:
            self._serial = device
            self._port = port
            self._output_path = output_path
            self._state = "recording"
            self._session = session
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"{self.kind}-recorder",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        device = self._serial
        if device is not None and self._shutdown_command is not None:
            try:
                device.write(self._shutdown_command)
                device.flush()
            except Exception:
                pass
        self._shutdown_command = None
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        device = self._serial
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        with self._lock:
            if self._state == "recording":
                self._state = "idle"
            self._serial = None
            self._thread = None

    def toggle(self) -> bool:
        if self.snapshot().state == "recording":
            self.stop()
            return True
        return self.start()

    def _record_loop(self) -> None:
        path = self.snapshot().output_path
        device = self._serial
        session = self._session
        if path is None or device is None or session is None:
            self._set_error("recorder was not initialized")
            return

        try:
            with path.open("x", encoding="utf-8", buffering=1) as handle:
                while not self._stop_event.is_set():
                    payload = self._read_payload(device)
                    if not payload:
                        continue
                    wall_time_ns = time.time_ns()
                    monotonic_ns = time.monotonic_ns()
                    if self.kind == "gps":
                        text = payload.decode("ascii", errors="replace").rstrip("\r\n")
                        match = NMEA_PATTERN.search(text)
                        if match is None:
                            continue
                        record = {
                            "session_id": session.session_id,
                            "session_start_wall_time_ns": session.start_wall_time_ns,
                            "session_start_monotonic_ns": session.start_monotonic_ns,
                            "wall_time_ns": wall_time_ns,
                            "monotonic_ns": monotonic_ns,
                            "elapsed_ns": monotonic_ns - session.start_monotonic_ns,
                            "port": self._port,
                            "nmea": text[match.start() :],
                        }
                    else:
                        record = {
                            "session_id": session.session_id,
                            "session_start_wall_time_ns": session.start_wall_time_ns,
                            "session_start_monotonic_ns": session.start_monotonic_ns,
                            "wall_time_ns": wall_time_ns,
                            "monotonic_ns": monotonic_ns,
                            "elapsed_ns": monotonic_ns - session.start_monotonic_ns,
                            "port": self._port,
                            "encoding": "base64",
                            "xbus": base64.b64encode(payload).decode("ascii"),
                        }
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    with self._lock:
                        self._records += 1
                        self._bytes_written += len(payload)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._set_error(str(exc))
        finally:
            try:
                device.close()
            except Exception:
                pass

    def _read_payload(self, device: Any) -> bytes:
        if self.kind == "gps":
            return bytes(device.readline())
        waiting = int(getattr(device, "in_waiting", 0) or 0)
        return bytes(device.read(min(max(waiting, 1), 4096)))

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = message


class SensorRecordControls:
    """One atomic GPS+IMU recording control rendered into an OpenCV BEV frame."""

    def __init__(self, gps: SerialRecorder, imu: SerialRecorder) -> None:
        self.recorders = (gps, imu)
        self.output_dir = gps.output_dir
        self.last_message = "Click to record GPS + IMU together"
        self._frame_width = 400
        self._active_session: RecordingSession | None = None

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._frame_width = frame_bgr.shape[1]
        gps, imu = (recorder.snapshot() for recorder in self.recorders)
        if self._active_session is not None:
            color = (20, 20, 210)
            text = f"STOP GPS + IMU  G:{gps.records} I:{imu.records}"
        elif gps.state == "error" or imu.state == "error":
            color = (0, 120, 220)
            text = "RETRY GPS + IMU"
        else:
            color = (65, 65, 65)
            text = "RECORD GPS + IMU"
        x1, y1, x2, y2 = self._button_bounds(self._frame_width)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 12, y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame_bgr, (0, 48), (self._frame_width, 70), (20, 20, 20), -1)
        cv2.putText(
            frame_bgr,
            self.last_message[:58],
            (8, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return frame_bgr

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONUP:
            return
        x1, y1, x2, y2 = self._button_bounds(self._frame_width)
        if x1 <= x <= x2 and y1 <= y <= y2:
            if self._active_session is None:
                self.start_session()
            else:
                self.stop_session()

    def active_session(self) -> RecordingSession | None:
        return self._active_session

    def start_session(self) -> bool:
        if self._active_session is not None:
            return True
        session = create_recording_session()
        gps, imu = self.recorders
        gps_started = gps.start(session)
        imu_started = imu.start(session)
        if not (gps_started and imu_started):
            gps.stop()
            imu.stop()
            errors = [
                snapshot.error
                for snapshot in (gps.snapshot(), imu.snapshot())
                if snapshot.error
            ]
            self.last_message = "; ".join(errors) or "Unable to start GPS + IMU"
            return False
        self._active_session = session
        self.last_message = f"Recording aligned session {session.session_id}"
        return True

    def stop_session(self) -> None:
        session = self._active_session
        if session is None:
            return
        end_wall_time_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        for recorder in self.recorders:
            recorder.stop()
        self._write_manifest(session, end_wall_time_ns, end_monotonic_ns)
        self._active_session = None
        self.last_message = f"Saved aligned session {session.session_id}"

    def stop_all(self) -> None:
        if self._active_session is not None:
            self.stop_session()
        else:
            for recorder in self.recorders:
                recorder.stop()

    def _write_manifest(
        self,
        session: RecordingSession,
        end_wall_time_ns: int,
        end_monotonic_ns: int,
    ) -> None:
        gps, imu = (recorder.snapshot() for recorder in self.recorders)
        manifest = {
            "session_id": session.session_id,
            "start_wall_time_ns": session.start_wall_time_ns,
            "start_monotonic_ns": session.start_monotonic_ns,
            "end_wall_time_ns": end_wall_time_ns,
            "end_monotonic_ns": end_monotonic_ns,
            "duration_ns": end_monotonic_ns - session.start_monotonic_ns,
            "clock": "time.monotonic_ns",
            "camera_log": "../detections.jsonl",
            "gps_log": gps.output_path.name if gps.output_path else None,
            "imu_log": imu.output_path.name if imu.output_path else None,
            "gps_records": gps.records,
            "imu_records": imu.records,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"session_{session.session_id}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _button_bounds(width: int) -> tuple[int, int, int, int]:
        return (8, 8, width - 8, 42)
