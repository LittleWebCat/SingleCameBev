from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Depth Anything V2 metric VKITTI Small to ONNX."
    )
    parser.add_argument("--checkpoint", required=True, help="Input .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--encoder", default="vits", choices=["vits"], help="Depth Anything V2 encoder")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    metric_depth_root = repo_root / "third_party" / "Depth-Anything-V2" / "metric_depth"
    sys.path.insert(0, str(metric_depth_root))

    from depth_anything_v2.dpt import DepthAnythingV2

    model_config = {
        "encoder": args.encoder,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "max_depth": args.max_depth,
    }
    model = DepthAnythingV2(**model_config)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["depth"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
