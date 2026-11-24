from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


class Da3MetricExport(nn.Module):
    """Single-view DA3 metric network with a fixed input size.

    Outputs the raw depth head (``depth``) and the sky head (``sky``). Convert
    to meters afterwards with ``depth * focal_px / 300``, where ``focal_px`` is
    the mean of fx and fy scaled to the network input resolution. Keeping the
    focal term out of the graph lets a new calibration take effect without
    rebuilding the engine.
    """

    def __init__(self, net: nn.Module, height: int, width: int) -> None:
        super().__init__()
        self.backbone = net.backbone
        self.head = net.head
        self.height = height
        self.width = width

        # DINOv2 resamples its positional embedding with bicubic interpolation on
        # every call. The input size is fixed, so compute it once and serve the
        # cached tensor; the exported graph then carries a constant instead of
        # a Resize that TensorRT may not match exactly.
        vit = self.backbone.pretrained
        patches = (height // vit.patch_size) * (width // vit.patch_size)
        tokens = torch.zeros(1, patches + 1, vit.embed_dim)
        with torch.no_grad():
            pos_embed = vit.interpolate_pos_encoding(tokens, height, width).float()
        self.register_buffer("pos_embed_fixed", pos_embed, persistent=False)
        vit.interpolate_pos_encoding = lambda x, w, h: self.pos_embed_fixed.to(x.dtype)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = images[:, None]  # (B, 3, H, W) -> (B, views=1, 3, H, W)
        feats, _ = self.backbone(x, export_feat_layers=[])
        out = self.head(feats, self.height, self.width, patch_start_idx=0)
        return out["depth"][:, 0], out["sky"][:, 0]


def load_da3_metric(checkpoint_dir: Path) -> nn.Module:
    from depth_anything_3.cfg import create_object
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    config = json.loads((checkpoint_dir / "config.json").read_text())
    net = create_object(OmegaConf.create(config["config"]))
    state = load_file(str(checkpoint_dir / "model.safetensors"))
    state = {key.removeprefix("model."): value for key, value in state.items()}
    net.load_state_dict(state, strict=True)
    return net.eval()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Depth Anything 3 Metric Large to ONNX for TensorRT."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="models/checkpoints/da3metric_large",
        help="Directory holding config.json and model.safetensors",
    )
    parser.add_argument("--output", default=None, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=378, help="Network input height")
    parser.add_argument("--width", type=int, default=504, help="Network input width")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    if args.height % 14 or args.width % 14:
        parser.error("--height and --width must be multiples of 14 (the ViT patch size)")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "third_party" / "Depth-Anything-3" / "src"))

    checkpoint_dir = Path(args.checkpoint_dir)
    output = Path(
        args.output or f"models/da3metric_large_{args.width}x{args.height}.onnx"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = Da3MetricExport(load_da3_metric(checkpoint_dir), args.height, args.width).eval()
    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    with torch.no_grad():
        depth, sky = model(dummy)
    print(f"PyTorch check: depth {tuple(depth.shape)}, sky {tuple(sky.shape)}")

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["images"],
        output_names=["depth", "sky"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
