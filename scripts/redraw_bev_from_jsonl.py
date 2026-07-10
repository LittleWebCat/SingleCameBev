from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bev_depth.config import BevConfig
from bev_depth.geometry import ego_to_bev_pixel
from bev_depth.io import save_bgr_image
from bev_depth.render import render_bev
from bev_depth.types import Detection, ProjectedDetection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw BEV frames from inference JSONL")
    parser.add_argument("--jsonl", required=True, help="Input detections.jsonl path")
    parser.add_argument("--out-dir", required=True, help="Output BEV image directory")
    parser.add_argument("--forward-range-m", type=float, default=80.0)
    parser.add_argument("--lateral-range-m", type=float, default=40.0)
    parser.add_argument("--pixels-per-meter", type=int, default=10)
    return parser.parse_args(argv)


def projected_from_record(record: dict, bev: BevConfig) -> list[ProjectedDetection]:
    projected: list[ProjectedDetection] = []
    for obj in record.get("objects", []):
        detection = Detection(
            class_name=str(obj["class_name"]),
            confidence=float(obj["confidence"]),
            bbox_xyxy=tuple(float(value) for value in obj["bbox_xyxy"]),
        )
        x_m = obj.get("x_m_lateral")
        z_m = obj.get("z_m_forward")
        bev_xy = None
        if x_m is not None and z_m is not None:
            bev_xy = ego_to_bev_pixel(float(x_m), float(z_m), bev)
        projected.append(
            ProjectedDetection(
                detection=detection,
                depth_m=obj.get("depth_m"),
                x_m_lateral=x_m,
                z_m_forward=z_m,
                bev_xy=bev_xy,
                depth_source=str(obj.get("depth_source", "")),
            )
        )
    return projected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bev = BevConfig(
        forward_range_m=args.forward_range_m,
        lateral_range_m=args.lateral_range_m,
        pixels_per_meter=args.pixels_per_meter,
    )
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame_id = int(record["frame_id"])
            image = render_bev(bev, projected_from_record(record, bev))
            save_bgr_image(out_dir / f"{frame_id:06d}.jpg", image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
