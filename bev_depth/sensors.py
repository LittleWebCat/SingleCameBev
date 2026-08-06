from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass
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
