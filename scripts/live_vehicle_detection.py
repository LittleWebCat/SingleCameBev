"""Live vehicle/road-user detection on selected cameras, shown side by side.

Uses the BEV project's YOLO TensorRT detector (person, bicycle, car,
motorcycle, bus, truck, stop sign) and CameraSetup/cameras.py for capture, so
cameras get the same mode, exposure and port names as the live viewer.

    cd ~/SingleCameraBev
    DISPLAY=:1 .venv/bin/python scripts/live_vehicle_detection.py --ports 4.1.2.1 4.1.2.3 4.1.3.3

Keys: o switches indoor/outdoor exposure, f toggles fullscreen, q quits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path.home() / "CameraSetup"))

from bev_depth.config import load_config  # noqa: E402
from bev_depth.detector import create_detector  # noqa: E402

import cameras  # noqa: E402

WINDOW = "Vehicle detection"
COLOURS = {
    "car": (0, 200, 255), "truck": (0, 120, 255), "bus": (0, 60, 255),
    "motorcycle": (255, 0, 200), "bicycle": (255, 120, 0), "person": (60, 255, 60),
    "stop sign": (0, 0, 255),
}


def draw(frame, detections, title, fps, infer_ms):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        colour = COLOURS.get(det.class_name, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                    cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, f"{title}   {fps:4.1f} fps   detect {infer_ms:4.1f} ms   {len(detections)} objects",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ports", nargs="+", required=True, help="USB ports, e.g. 4.1.2.1 4.1.2.3")
    parser.add_argument("--config", default=str(REPO / "configs/three_camera_bev.yaml"),
                        help="config whose detector section is used")
    parser.add_argument("--confidence", type=float, help="override the detection threshold")
    parser.add_argument("--outdoor", action="store_true", help="start with auto exposure")
    args = parser.parse_args()

    if args.outdoor:
        cameras.LIGHTING = "outdoor"
    config = load_config(args.config)
    detector_cfg = config.detector
    if args.confidence is not None:
        from dataclasses import replace
        detector_cfg = replace(detector_cfg, confidence=args.confidence)
    print(f"Loading detector {detector_cfg.path} ...")
    detector = create_detector(detector_cfg)

    found = cameras.find_cameras()
    missing = [p for p in args.ports if p not in found]
    if missing:
        print(f"Not connected: {', '.join(missing)}")
    grabbers = []
    for port in args.ports:
        if port in found:
            grabber = cameras.Grabber(found[port], port)
            grabber.start()
            grabbers.append(grabber)
            time.sleep(cameras.OPEN_STAGGER)
    if not grabbers:
        return 1

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = False
    last_stamp = {g.port: 0.0 for g in grabbers}
    tiles = {g.port: cameras.placeholder("STARTING", g.label) for g in grabbers}
    infer_ms = {g.port: 0.0 for g in grabbers}
    try:
        while True:
            for g in grabbers:
                if g.last_frame_at == last_stamp[g.port]:
                    continue                      # no new frame from this camera yet
                last_stamp[g.port] = g.last_frame_at
                frame = g.read()
                if frame is None:
                    continue
                t0 = time.perf_counter()
                detections = detector.infer(frame)
                ms = (time.perf_counter() - t0) * 1000
                infer_ms[g.port] = 0.9 * infer_ms[g.port] + 0.1 * ms if infer_ms[g.port] else ms
                title = f"{cameras.ROLES.get(g.port, g.name)} ({g.port})"
                tiles[g.port] = cv2.resize(
                    draw(frame, detections, title, g.fps, infer_ms[g.port]), (640, 480))
            cv2.imshow(WINDOW, np.hstack([tiles[g.port] for g in grabbers]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                cameras.set_lighting(grabbers, "indoor" if cameras.LIGHTING == "outdoor" else "outdoor")
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        pass
    finally:
        for g in grabbers:
            g.stop()
        for g in grabbers:
            g.join(timeout=1.0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
