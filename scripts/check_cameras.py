"""Find and test every connected USB camera.

For each camera: USB port, /dev/video node, name, the mode it actually runs,
frames received, measured FPS and exposure mode. Then all cameras stream at
once to catch shared-bandwidth problems. It also lists USB ports the kernel
failed to enumerate, flags cameras a BEV config expects but cannot find, and
saves one labelled snapshot sheet.

    .venv/bin/python scripts/check_cameras.py
    .venv/bin/python scripts/check_cameras.py --width 1920 --height 1200 --fps 30
    .venv/bin/python scripts/check_cameras.py --config configs/three_camera_bev.yaml
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_cameras() -> dict[str, str]:
    """USB port -> capture node. by-path names survive /dev/videoN renumbering;
    the same camera can appear as usb-0 and usbv2-0, so dedupe by port."""
    found: dict[str, str] = {}
    for path in sorted(glob.glob("/dev/v4l/by-path/*video-index0")):
        match = re.search(r"-0:([0-9.]+):", path)
        port = match.group(1) if match else path
        found.setdefault(port, path)
    return found


def camera_name(path: str) -> str:
    node = os.path.basename(os.path.realpath(path))
    try:
        name = Path(f"/sys/class/video4linux/{node}/name").read_text().strip()
    except OSError:
        return "?"
    return name.split(":")[0].strip() or name


def v4l2(path: str, *args: str) -> str:
    if not shutil.which("v4l2-ctl"):
        return ""
    try:
        return subprocess.run(
            ["v4l2-ctl", "-d", path, *args], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return ""


def exposure_mode(path: str) -> str:
    out = v4l2(path, "--get-ctrl=auto_exposure")
    if "Manual" in out or ": 1" in out:
        exposure = v4l2(path, "--get-ctrl=exposure_time_absolute").split(":")[-1].strip()
        return f"manual {exposure}"
    return "auto" if out else "?"


def open_camera(path: str, args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    # MJPG before width/height, or the driver negotiates raw YUYV.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)  # 1 buffer halves the frame rate
    return cap


def measure(cap: cv2.VideoCapture, seconds: float) -> tuple[int, float, np.ndarray | None]:
    """Frames and FPS over `seconds` after a short warm-up; gives up on a dead stream."""
    frame = None
    for _ in range(5):
        ok, f = cap.read()
        if ok:
            frame = f
    count, start = 0, time.time()
    while time.time() - start < seconds:
        ok, f = cap.read()
        if not ok:
            if count == 0 and time.time() - start > 2.0:
                break
            continue
        count += 1
        frame = f
    return count, count / seconds, frame


def actual_mode(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"{fourcc}"
    )


def failed_ports(minutes: int) -> list[str]:
    """USB ports the kernel could not enumerate recently (bad cable, power, port)."""
    try:
        log = subprocess.run(
            ["journalctl", "-k", "--since", f"-{minutes} min", "--no-pager"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    ports = set(re.findall(r"usb (\d+-[\d.]+)-port(\d+): unable to enumerate", log))
    return sorted(f"{hub.split('-', 1)[1]}.{port}" for hub, port in ports)


def configured_ports(config_path: str) -> dict[str, str]:
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    ports = {}
    for cam in raw.get("cameras", []) or []:
        match = re.search(r"-0:([0-9.]+):", str(cam.get("source", "")))
        if match:
            ports[match.group(1)] = cam.get("name", "?")
    return ports


def tile(frame: np.ndarray | None, text: str) -> np.ndarray:
    image = np.zeros((360, 480, 3), np.uint8) if frame is None else cv2.resize(frame, (480, 360))
    if frame is None:
        cv2.putText(image, "NO FRAMES", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 230), 2)
    cv2.rectangle(image, (0, 0), (480, 28), (0, 0, 0), -1)
    cv2.putText(image, text[:52], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--seconds", type=float, default=3.0, help="Measurement time per test")
    parser.add_argument("--no-together", action="store_true", help="Skip the all-at-once test")
    parser.add_argument("--config", help="BEV config whose camera ports should be present")
    parser.add_argument("--out", default="runs/camera_check_now/cameras.jpg")
    args = parser.parse_args()

    cameras = find_cameras()
    print(f"Found {len(cameras)} camera(s); requesting {args.width}x{args.height} "
          f"{args.fourcc} @ {args.fps}\n")

    results = {}
    print(f"{'port':10} {'node':8} {'name':22} {'mode':16} {'frames':>6} {'fps':>6}  exposure")
    for port, path in cameras.items():
        cap = open_camera(path, args)
        if not cap.isOpened():
            print(f"{port:10} {'-':8} {camera_name(path):22} CANNOT OPEN")
            results[port] = (0, 0.0, None)
            continue
        count, fps, frame = measure(cap, args.seconds)
        mode = actual_mode(cap)
        cap.release()
        results[port] = (count, fps, frame)
        flag = "" if count else "   <-- NO FRAMES"
        if count and fps < 0.8 * args.fps:
            flag = "   <-- slow (try manual exposure or more light)"
        node = os.path.basename(os.path.realpath(path))
        print(f"{port:10} {node:8} {camera_name(path):22} {mode:16} {count:6d} {fps:6.1f}  "
              f"{exposure_mode(path)}{flag}")

    if len(cameras) > 1 and not args.no_together:
        print("\nAll cameras streaming together:")
        caps = {port: open_camera(path, args) for port, path in cameras.items()}
        together: dict[str, float] = {}

        def run(port: str) -> None:
            together[port] = measure(caps[port], args.seconds)[1]

        threads = [threading.Thread(target=run, args=(port,)) for port in caps]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for port, cap in caps.items():
            cap.release()
            alone = results[port][1]
            drop = "   <-- drops when shared (USB bandwidth)" if together[port] < 0.8 * alone else ""
            print(f"  {port:10} {together[port]:6.1f} fps (alone {alone:.1f}){drop}")

    bad = failed_ports(30)
    if bad:
        print("\nUSB ports that failed to enumerate in the last 30 min "
              "(reseat the cable, try another cable/port, check hub power):")
        for port in bad:
            note = "  (connected now; it failed earlier, so check the cable)" if port in cameras else ""
            print(f"  {port}{note}")

    if args.config:
        expected = configured_ports(args.config)
        missing = {p: n for p, n in expected.items() if p not in cameras}
        print(f"\n{args.config}:")
        for port, name in expected.items():
            state = "MISSING" if port in missing else "ok"
            print(f"  {name:8} port {port:10} {state}")

    if results:
        tiles = [
            tile(frame, f"{port}  {camera_name(cameras[port])}  {fps:.0f} fps")
            for port, (count, fps, frame) in results.items()
        ]
        cols = min(3, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print(f"\nSnapshot sheet: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
