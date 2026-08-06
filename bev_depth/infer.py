from __future__ import annotations

import argparse
import atexit
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

from .config import load_config
from .depth import create_depth_backend, sample_detection_depth
from .detector import create_detector
from .geometry import ego_to_bev_pixel, pixel_depth_to_ego, scale_intrinsics
from .io import ensure_output_dirs, iter_frames, save_bgr_image, write_jsonl_record
from .profiler import Profiler
from .render import draw_annotated_frame, render_bev, render_depth
from .sensors import SensorRecordControls, SerialRecorder
from .types import ProjectedDetection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera OD + depth + BEV inference")
    parser.add_argument("--source", required=True, help="Camera index, image, directory, or video path")
    parser.add_argument("--config", default="configs/thor_1080p.yaml", help="YAML config path")
    parser.add_argument("--save-dir", default="runs/bev_depth", help="Output directory")
    parser.add_argument("--display", action="store_true", help="Show live annotated and BEV preview")
    parser.add_argument("--display-bev", action="store_true", help="Show only the live BEV preview")
    parser.add_argument("--no-save-images", action="store_true", help="Write JSONL only")
    parser.add_argument("--save-bev-only", action="store_true", help="Save BEV images without annotated/depth images")
    parser.add_argument("--max-frames", type=int, default=None, help="Override config runtime.max_frames")
    parser.add_argument("--input-width", type=int, default=None, help="Override camera capture width")
    parser.add_argument("--input-height", type=int, default=None, help="Override camera capture height")
    parser.add_argument("--input-fps", type=int, default=None, help="Override camera capture FPS")
    parser.add_argument("--input-fourcc", default=None, help="Override camera capture FOURCC, for example MJPG")
    parser.add_argument(
        "--gps-port",
        default="auto",
        help="GPS serial port or 'auto' (default: Septentrio USB1/ttyACM0)",
    )
    parser.add_argument(
        "--imu-port",
        default="auto",
        help="IMU serial port or 'auto' (default: Xsens/ttyUSB0)",
    )
    parser.add_argument("--gps-baud", type=int, default=115200)
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--profile", action="store_true", help="Print stage latency summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    input_overrides = {}
    if args.input_width is not None:
        input_overrides["width"] = args.input_width
    if args.input_height is not None:
        input_overrides["height"] = args.input_height
    if args.input_fps is not None:
        input_overrides["fps"] = args.input_fps
    if args.input_fourcc is not None:
        input_overrides["fourcc"] = args.input_fourcc
    if input_overrides:
        config = replace(config, input=replace(config.input, **input_overrides))
    dirs = ensure_output_dirs(args.save_dir)
    detector = create_detector(config.detector, config.traffic_detector)
    depth_backend = create_depth_backend(config.depth)
    profiler = Profiler()
    latest_depth = None
    processed_frames = 0
    loop_start = time.perf_counter()
    sensor_controls = None
    if args.display or args.display_bev:
        sensor_dir = dirs["root"] / "sensors"
        sensor_controls = SensorRecordControls(
            gps=SerialRecorder("gps", sensor_dir, args.gps_port, args.gps_baud),
            imu=SerialRecorder("imu", sensor_dir, args.imu_port, args.imu_baud),
        )
        cv2.namedWindow("bev", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("bev", sensor_controls.mouse_callback)
        atexit.register(sensor_controls.stop_all)

    jsonl_path = Path(args.save_dir) / "detections.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for frame_id, timestamp, frame in iter_frames(args.source, config.input):
            frame_wall_time_ns = time.time_ns()
            frame_monotonic_ns = time.monotonic_ns()
            frame_height, frame_width = frame.shape[:2]
            frame_intrinsics = scale_intrinsics(
                config.intrinsics, frame_width, frame_height
            )
            sensor_session = (
                sensor_controls.active_session() if sensor_controls is not None else None
            )
            max_frames = args.max_frames if args.max_frames is not None else config.runtime.max_frames
            if max_frames and frame_id >= max_frames:
                break

            with profiler.stage("detect"):
                detections = detector.infer(frame)

            run_depth = latest_depth is None or frame_id % config.depth.cadence == 0
            if run_depth:
                with profiler.stage("depth"):
                    latest_depth = depth_backend.infer(frame)
                depth_source = "depth_map"
            else:
                depth_source = "reused_depth_map"

            projected: list[ProjectedDetection] = []
            with profiler.stage("project"):
                for detection in detections:
                    depth_m, source = sample_detection_depth(latest_depth, detection, config.depth)
                    if depth_m is None:
                        projected.append(
                            ProjectedDetection(detection, None, None, None, None, source)
                        )
                        continue
                    x1, y1, x2, y2 = detection.bbox_xyxy
                    u = (x1 + x2) * 0.5
                    v = y2
                    x_m, z_m = pixel_depth_to_ego(
                        u, v, depth_m, frame_intrinsics, config.extrinsics
                    )
                    bev_xy = ego_to_bev_pixel(x_m, z_m, config.bev)
                    projected.append(
                        ProjectedDetection(
                            detection=detection,
                            depth_m=depth_m,
                            x_m_lateral=x_m,
                            z_m_forward=z_m,
                            bev_xy=bev_xy,
                            depth_source=depth_source if source == "depth_map" else source,
                        )
                    )

            write_jsonl_record(
                jsonl,
                frame_id,
                timestamp,
                projected,
                wall_time_ns=frame_wall_time_ns,
                monotonic_ns=frame_monotonic_ns,
                sensor_session_id=(
                    sensor_session.session_id if sensor_session is not None else None
                ),
                sensor_elapsed_ns=(
                    frame_monotonic_ns - sensor_session.start_monotonic_ns
                    if sensor_session is not None
                    else None
                ),
            )

            save_images = config.runtime.save_images and not args.no_save_images
            display_any = args.display or args.display_bev
            if save_images or display_any:
                with profiler.stage("render"):
                    annotated = None
                    depth_vis = None
                    need_full_render = args.display or (save_images and not args.save_bev_only)
                    if need_full_render:
                        annotated = draw_annotated_frame(frame, projected)
                        depth_vis = render_depth(latest_depth)
                    bev = render_bev(config.bev, projected)
                    display_bev = (
                        sensor_controls.draw(bev.copy()) if sensor_controls is not None else bev
                    )
                if save_images:
                    stem = f"{frame_id:06d}.jpg"
                    if not args.save_bev_only:
                        save_bgr_image(dirs["annotated"] / stem, annotated)
                        save_bgr_image(dirs["depth"] / stem, depth_vis)
                    save_bgr_image(dirs["bev"] / stem, bev)
                if args.display:
                    cv2.imshow("annotated", annotated)
                    cv2.imshow("depth", depth_vis)
                    cv2.imshow("bev", display_bev)
                    if cv2.waitKey(1) == 27:
                        break
                elif args.display_bev:
                    cv2.imshow("bev", display_bev)
                    if cv2.waitKey(1) == 27:
                        break
            processed_frames += 1

    if sensor_controls is not None:
        sensor_controls.stop_all()
        atexit.unregister(sensor_controls.stop_all)
    if args.display or args.display_bev:
        cv2.destroyAllWindows()
    if args.profile:
        elapsed = time.perf_counter() - loop_start
        print(profiler.summary())
        print(
            f"frames={processed_frames} elapsed={elapsed:.2f}s "
            f"end_to_end_fps={processed_frames / elapsed if elapsed > 0.0 else 0.0:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
