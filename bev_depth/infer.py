from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import load_config
from .depth import create_depth_backend, sample_detection_depth
from .detector import create_detector
from .geometry import ego_to_bev_pixel, pixel_depth_to_ego
from .io import ensure_output_dirs, iter_frames, save_bgr_image, write_jsonl_record
from .profiler import Profiler
from .render import draw_annotated_frame, render_bev, render_depth
from .types import ProjectedDetection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera OD + depth + BEV inference")
    parser.add_argument("--source", required=True, help="Camera index, image, directory, or video path")
    parser.add_argument("--config", default="configs/thor_1080p.yaml", help="YAML config path")
    parser.add_argument("--save-dir", default="runs/bev_depth", help="Output directory")
    parser.add_argument("--display", action="store_true", help="Show live annotated and BEV preview")
    parser.add_argument("--no-save-images", action="store_true", help="Write JSONL only")
    parser.add_argument("--profile", action="store_true", help="Print stage latency summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    dirs = ensure_output_dirs(args.save_dir)
    detector = create_detector(config.detector)
    depth_backend = create_depth_backend(config.depth)
    profiler = Profiler()
    latest_depth = None

    jsonl_path = Path(args.save_dir) / "detections.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for frame_id, timestamp, frame in iter_frames(args.source):
            if config.runtime.max_frames and frame_id >= config.runtime.max_frames:
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
                        u, v, depth_m, config.intrinsics, config.extrinsics
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

            write_jsonl_record(jsonl, frame_id, timestamp, projected)

            save_images = config.runtime.save_images and not args.no_save_images
            if save_images or args.display:
                with profiler.stage("render"):
                    annotated = draw_annotated_frame(frame, projected)
                    depth_vis = render_depth(latest_depth)
                    bev = render_bev(config.bev, projected)
                if save_images:
                    stem = f"{frame_id:06d}.jpg"
                    save_bgr_image(dirs["annotated"] / stem, annotated)
                    save_bgr_image(dirs["depth"] / stem, depth_vis)
                    save_bgr_image(dirs["bev"] / stem, bev)
                if args.display:
                    cv2.imshow("annotated", annotated)
                    cv2.imshow("depth", depth_vis)
                    cv2.imshow("bev", bev)
                    if cv2.waitKey(1) == 27:
                        break

    if args.display:
        cv2.destroyAllWindows()
    if args.profile:
        print(profiler.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
