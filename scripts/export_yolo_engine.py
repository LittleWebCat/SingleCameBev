from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO model to TensorRT engine.")
    parser.add_argument("--weights", required=True, help="Input YOLO .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quantize", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--data", default=None, help="Dataset YAML for INT8 calibration")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": 1,
        "dynamic": False,
        "simplify": True,
    }
    if args.quantize in (8, 16):
        kwargs["quantize"] = args.quantize
    if args.data:
        kwargs["data"] = args.data
    model.export(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
