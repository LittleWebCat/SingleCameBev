"""Compare Depth Anything V2 (current) and Depth Anything 3 Metric engines.

Speed: both TensorRT engines run on the same frames, timed per stage the way
the live pipeline uses them (preprocess, GPU inference, resize back to the
frame).

Accuracy: pass --gt-csv with measured distances to score both models. Without
it the script still reports how far apart the two models' distances are and
saves side-by-side depth images on a shared color scale.

Run inside .venv-da3 (the DA3 PyTorch reference needs its extra packages):

    .venv-da3/bin/python scripts/benchmark_depth_da2_vs_da3.py \
        --images runs/three_camera_preview/camera*.jpg
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bev_depth.config import load_config  # noqa: E402
from bev_depth.depth import _depth_preprocess  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DA3_FOCAL_DIVISOR = 300.0  # metric = raw * focal_px / 300 (DA3 README)
DA3_SKY_THRESHOLD = 0.3  # same threshold the DA3 code uses for its sky mask
VIS_MAX_DEPTH_M = 80.0


class TrtRunner:
    """Minimal TensorRT executor with fixed shapes and named outputs."""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart

        self.trt = trt
        self.cudart = cudart
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]

        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype] = {}
        self.device: dict[str, int] = {}
        self.host: dict[str, np.ndarray] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"{engine_path} has a dynamic dimension on {name}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            err, ptr = cudart.cudaMalloc(int(np.prod(shape)) * dtype.itemsize)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
            self.device[name] = ptr
            self.context.set_tensor_address(name, ptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
                self.host[name] = np.empty(shape, dtype=dtype)
        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected one input tensor, got {self.inputs}")

    @property
    def input_hw(self) -> tuple[int, int]:
        shape = self.shapes[self.inputs[0]]
        return int(shape[-2]), int(shape[-1])

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        name = self.inputs[0]
        tensor = np.ascontiguousarray(tensor.astype(self.dtypes[name], copy=False))
        cudart = self.cudart
        cudart.cudaMemcpyAsync(
            self.device[name],
            tensor.ctypes.data,
            tensor.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        for out in self.outputs:
            cudart.cudaMemcpyAsync(
                self.host[out].ctypes.data,
                self.device[out],
                self.host[out].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        cudart.cudaStreamSynchronize(self.stream)
        return {out: self.host[out].copy() for out in self.outputs}


def da3_preprocess(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match DA3's InputProcessor: RGB, area/cubic resize, ImageNet normalize."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    upscale = width > frame_bgr.shape[1] or height > frame_bgr.shape[0]
    interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
    resized = cv2.resize(rgb, (width, height), interpolation=interp)
    image = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return np.transpose(image, (2, 0, 1))[None, ...]


class Da2Model:
    name = "DA2 metric-vkitti Small"

    def __init__(self, engine: Path, input_size: int, scale: float) -> None:
        self.runner = TrtRunner(engine)
        self.input_size = input_size
        self.scale = scale

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return _depth_preprocess(frame, self.input_size)

    def postprocess(self, outputs: dict[str, np.ndarray], frame: np.ndarray):
        h, w = frame.shape[:2]
        depth = next(iter(outputs.values())).squeeze().astype(np.float32)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC) * self.scale
        return depth, None


class Da3Model:
    name = "DA3 Metric Large"

    def __init__(self, engine: Path, focal_net_px: float, scale: float) -> None:
        self.runner = TrtRunner(engine)
        self.height, self.width = self.runner.input_hw
        self.focal_net_px = focal_net_px
        self.scale = scale

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return da3_preprocess(frame, self.height, self.width)

    def to_meters(self, raw: np.ndarray) -> np.ndarray:
        return raw * (self.focal_net_px / DA3_FOCAL_DIVISOR) * self.scale

    def postprocess(self, outputs: dict[str, np.ndarray], frame: np.ndarray):
        h, w = frame.shape[:2]
        raw = outputs["depth"].squeeze().astype(np.float32)
        sky = outputs["sky"].squeeze().astype(np.float32)
        depth = cv2.resize(self.to_meters(raw), (w, h), interpolation=cv2.INTER_CUBIC)
        sky = cv2.resize(sky, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth, sky >= DA3_SKY_THRESHOLD


def load_frames(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    for pattern in args.images or []:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            image = cv2.imread(path)
            if image is None:
                raise SystemExit(f"Cannot read image: {path}")
            frames.append((path, image))
    if args.video:
        cap = cv2.VideoCapture(args.video)
        index = 0
        while len(frames) < args.max_frames:
            ok, image = cap.read()
            if not ok:
                break
            if index % args.video_every == 0:
                frames.append((f"{args.video}#{index}", image))
            index += 1
        cap.release()
    if not frames:
        raise SystemExit("No frames: pass --images and/or --video")
    return frames


def timed_run(model, frame: np.ndarray):
    t0 = time.perf_counter()
    tensor = model.preprocess(frame)
    t1 = time.perf_counter()
    outputs = model.runner.run(tensor)
    t2 = time.perf_counter()
    depth, sky = model.postprocess(outputs, frame)
    t3 = time.perf_counter()
    return depth, sky, outputs, ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"mean": statistics.fmean(values), "p50": statistics.median(values), "p95": p95}


def benchmark_speed(
    model, frames, warmup: int, iterations: int, pace_s: float = 0.0
) -> dict[str, dict[str, float]]:
    for i in range(warmup):
        timed_run(model, frames[i % len(frames)][1])
        time.sleep(pace_s)
    stages: dict[str, list[float]] = {"preprocess": [], "gpu": [], "postprocess": [], "total": []}
    for i in range(iterations):
        _, _, _, (pre, gpu, post) = timed_run(model, frames[i % len(frames)][1])
        stages["preprocess"].append(pre)
        stages["gpu"].append(gpu)
        stages["postprocess"].append(post)
        stages["total"].append(pre + gpu + post)
        time.sleep(pace_s)
    return {stage: summarize(values) for stage, values in stages.items()}


def roi_median(depth: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float | None:
    """Same box region the live pipeline samples (bev_depth.depth.sample_detection_depth)."""
    h, w = depth.shape[:2]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx1 = int(np.clip(x1 + 0.35 * bw, 0, w - 1))
    cx2 = int(np.clip(x1 + 0.65 * bw, 0, w))
    cy1 = int(np.clip(y1 + 0.65 * bh, 0, h - 1))
    cy2 = int(np.clip(y1 + 0.9 * bh, 0, h))
    roi = depth[cy1:cy2, cx1:cx2]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    return float(np.median(roi)) if roi.size else None


def point_median(depth: np.ndarray, x: float, y: float, radius: int = 3) -> float | None:
    h, w = depth.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    patch = depth[max(0, yi - radius) : yi + radius + 1, max(0, xi - radius) : xi + radius + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size and 0 <= xi < w and 0 <= yi < h else None


def score_ground_truth(gt_csv: Path, frames, results) -> dict:
    """GT rows: image,x,y,distance_m  or  image,x1,y1,x2,y2,distance_m (pixels of the original image)."""
    by_name = {name: i for i, (name, _) in enumerate(frames)}
    by_base = {Path(name).name: i for i, (name, _) in enumerate(frames)}
    rows_out, errors = [], {key: [] for key in results}
    with gt_csv.open() as handle:
        for row in csv.DictReader(handle):
            index = by_name.get(row["image"], by_base.get(Path(row["image"]).name))
            if index is None:
                print(f"  GT row skipped, image not in the frame set: {row['image']}")
                continue
            truth = float(row["distance_m"])
            record = {"image": row["image"], "distance_m": truth}
            for key, per_frame in results.items():
                depth = per_frame[index]
                if row.get("x1"):
                    pred = roi_median(
                        depth, *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                    )
                else:
                    pred = point_median(depth, float(row["x"]), float(row["y"]))
                record[key] = pred
                if pred is not None:
                    errors[key].append((pred, truth))
            rows_out.append(record)

    summary = {}
    for key, pairs in errors.items():
        if not pairs:
            continue
        preds = np.array([p for p, _ in pairs])
        truth = np.array([t for _, t in pairs])
        summary[key] = {
            "points": len(pairs),
            "mae_m": float(np.mean(np.abs(preds - truth))),
            "abs_rel": float(np.mean(np.abs(preds - truth) / truth)),
            "within_10pct": float(np.mean(np.abs(preds - truth) / truth < 0.10)),
            # Multiply the model's output by this to best match the tape measure.
            "best_fit_scale": float(np.median(truth / preds)),
        }
    return {"rows": rows_out, "summary": summary}


def colorize(
    depth: np.ndarray, max_m: float, invalid: np.ndarray | None = None
) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(depth, nan=max_m), 0, max_m)
    # Near = bright, far = dark. Both models share max_m so colors compare directly.
    scaled = (255 * (1.0 - clipped / max_m)).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    if invalid is not None:
        image[invalid] = (90, 60, 40)
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", nargs="*", help="Image files or glob patterns")
    parser.add_argument("--video", help="Video file to sample frames from")
    parser.add_argument("--video-every", type=int, default=50, help="Keep every Nth video frame")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--config", default="configs/three_camera_bev.yaml",
                        help="Pipeline config: DA2 engine, metric_scale, and camera intrinsics")
    parser.add_argument("--da2-engine", help="Override the DA2 engine from --config")
    parser.add_argument("--da2-scale", type=float,
                        help="Multiply DA2 output (default: metric_scale from --config)")
    parser.add_argument("--da3-engine", default="models/da3metric_large_504x378_fp16.engine")
    parser.add_argument("--da3-scale", type=float, default=1.0,
                        help="Extra multiplier on DA3 meters (leave 1.0 to test it as shipped)")
    parser.add_argument("--fx", type=float, help="Override fx (pixels at calibration size)")
    parser.add_argument("--fy", type=float, help="Override fy (pixels at calibration size)")
    parser.add_argument("--gt-csv", type=Path, help="Measured distances to score against")
    parser.add_argument("--torch-reference", action="store_true",
                        help="Also run DA3 in PyTorch fp32 to check the TensorRT engine's error")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pace-ms", type=float, default=0.0,
                        help="Pause after each inference; lowers average power on a weak supply")
    parser.add_argument("--cameras", type=int, default=3, help="Cameras in the live pipeline")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/depth_benchmark_da2_vs_da3"))
    args = parser.parse_args()

    config = load_config(args.config)
    intr = config.intrinsics
    fx = args.fx if args.fx is not None else intr.fx
    fy = args.fy if args.fy is not None else intr.fy
    frames = load_frames(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    da2 = Da2Model(
        Path(args.da2_engine or config.depth.path),
        config.depth.input_size,
        args.da2_scale if args.da2_scale is not None else config.depth.metric_scale,
    )
    da3 = Da3Model(Path(args.da3_engine), 0.0, args.da3_scale)
    # DA3 needs the focal length in pixels of ITS input image, so scale the
    # calibration-size focal length to the network resolution.
    focal_net = 0.5 * (
        fx * da3.width / intr.calibration_width + fy * da3.height / intr.calibration_height
    )
    da3.focal_net_px = focal_net

    print(f"Frames: {len(frames)}  (first: {frames[0][0]}, {frames[0][1].shape[1]}x{frames[0][1].shape[0]})")
    print(f"{da2.name}: input {da2.input_size}x{da2.input_size}, scale x{da2.scale:g}")
    print(f"{da3.name}: input {da3.width}x{da3.height}, fx={fx:g} fy={fy:g} at "
          f"{intr.calibration_width}x{intr.calibration_height} -> focal {focal_net:.1f} px "
          f"at network size, scale x{da3.scale:g}")

    pace_s = args.pace_ms / 1000.0
    summary = {
        "frames": [name for name, _ in frames],
        "da2": {"engine": str(args.da2_engine or config.depth.path), "scale": da2.scale,
                "input": [da2.input_size, da2.input_size]},
        "da3": {"engine": args.da3_engine, "input": [da3.width, da3.height],
                "fx": fx, "fy": fy, "focal_net_px": focal_net, "scale": da3.scale},
        "pace_ms": args.pace_ms,
    }

    def save() -> None:
        # Written after every stage and synced, so results survive a sudden reset.
        path = args.out_dir / "summary.json"
        with path.open("w") as handle:
            handle.write(json.dumps(summary, indent=2))
            handle.flush()
            os.fsync(handle.fileno())

    # ---- depth maps ----------------------------------------------------------
    maps = {"da2": [], "da3": []}
    sky_masks, raw_da3 = [], []
    for name, frame in frames:
        d2, _, _, _ = timed_run(da2, frame)
        time.sleep(pace_s)
        d3, sky, out3, _ = timed_run(da3, frame)
        time.sleep(pace_s)
        maps["da2"].append(d2)
        maps["da3"].append(d3)
        sky_masks.append(sky)
        raw_da3.append(out3["depth"].squeeze().astype(np.float32))
    print(f"Depth maps done for {len(frames)} frames", flush=True)

    # How far apart the two models are, on ground pixels (lower 60%, not sky).
    ratios = []
    for d2, d3, sky in zip(maps["da2"], maps["da3"], sky_masks):
        mask = np.zeros(d2.shape, bool)
        mask[int(0.4 * d2.shape[0]) :] = True
        mask &= ~sky & (d2 > 0.5) & (d3 > 0.5)
        if mask.any():
            ratios.append(float(np.median(d3[mask] / d2[mask])))
    agreement = {
        "median_da3_over_da2": statistics.median(ratios) if ratios else None,
        "per_frame": ratios,
    }
    if ratios:
        print(f"\nDA3 / DA2 distance ratio (ground pixels, median over frames): "
              f"{agreement['median_da3_over_da2']:.2f}  "
              f"[min {min(ratios):.2f}, max {max(ratios):.2f}]")

    # ---- TensorRT vs PyTorch fp32 --------------------------------------------
    reference = None
    if args.torch_reference:
        import torch

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from export_depth_anything3_metric_onnx import Da3MetricExport, load_da3_metric

        net = load_da3_metric(REPO_ROOT / "models" / "checkpoints" / "da3metric_large")
        module = Da3MetricExport(net, da3.height, da3.width).cuda().eval()
        rel_errors = []
        with torch.inference_mode():
            for (name, frame), trt_raw in zip(frames, raw_da3):
                tensor = torch.from_numpy(da3.preprocess(frame)).cuda()
                ref = module(tensor)[0].squeeze().float().cpu().numpy()
                rel_errors.append(float(np.median(np.abs(trt_raw - ref) / np.maximum(ref, 1e-3))))
        reference = {"median_rel_error": statistics.median(rel_errors), "per_frame": rel_errors}
        print(f"\nDA3 TensorRT vs PyTorch fp32: median relative difference "
              f"{reference['median_rel_error'] * 100:.2f}%  (max frame "
              f"{max(rel_errors) * 100:.2f}%)")
        del module, net
        torch.cuda.empty_cache()

    # ---- ground truth --------------------------------------------------------
    gt = None
    if args.gt_csv:
        gt = score_ground_truth(args.gt_csv, frames, maps)
        print("\nAgainst measured distances:")
        for key, stats in gt["summary"].items():
            model = da2 if key == "da2" else da3
            print(f"  {model.name:26} points={stats['points']:3d}  MAE={stats['mae_m']:.2f} m  "
                  f"AbsRel={stats['abs_rel'] * 100:.1f}%  within10%={stats['within_10pct'] * 100:.0f}%  "
                  f"best-fit scale x{stats['best_fit_scale']:.3f}")

    # ---- images --------------------------------------------------------------
    for i, (name, frame) in enumerate(frames):
        # Shared color range per frame: up to the farther model's 95th percentile,
        # capped at the 80 m DA2 limit, so indoor and road scenes both stay readable.
        both = np.concatenate([maps["da2"][i].ravel(), maps["da3"][i][~sky_masks[i]].ravel()])
        max_m = float(np.clip(np.percentile(both, 95), 2.0, VIS_MAX_DEPTH_M))
        panels = [
            label(frame, Path(name).name),
            label(colorize(maps["da2"][i], max_m), f"DA2 small x{da2.scale:g}  (0-{max_m:.1f} m)"),
            label(colorize(maps["da3"][i], max_m, sky_masks[i]),
                  f"DA3 metric large  (0-{max_m:.1f} m, sky = blue)"),
        ]
        cv2.imwrite(str(args.out_dir / f"compare_{i:03d}.jpg"), np.hstack(panels))
        np.savez_compressed(
            args.out_dir / f"depth_{i:03d}.npz",
            da2=maps["da2"][i], da3=maps["da3"][i], da3_sky=sky_masks[i],
        )
    os.sync()
    summary.update({"agreement": agreement, "torch_reference": reference, "ground_truth": gt})
    save()
    print(f"Saved {len(frames)} comparison images and accuracy results", flush=True)
    if args.iterations <= 0:
        return 0

    # ---- speed ---------------------------------------------------------------
    speed = {}
    summary["speed_ms"] = speed
    for key, model in (("da2", da2), ("da3", da3)):
        print(f"Timing {model.name} ({args.iterations} runs, {args.pace_ms:g} ms pause)...",
              flush=True)
        speed[key] = benchmark_speed(model, frames, args.warmup, args.iterations, pace_s)
        save()
    print(f"\nSpeed over {args.iterations} runs (ms; mean / p50 / p95)")
    print(f"{'':26}{'preprocess':>20}{'GPU':>20}{'postprocess':>20}{'total':>20}")
    for key, model in (("da2", da2), ("da3", da3)):
        cells = "".join(
            f"{speed[key][s]['mean']:8.2f}/{speed[key][s]['p50']:5.2f}/{speed[key][s]['p95']:5.2f}"
            for s in ("preprocess", "gpu", "postprocess", "total")
        )
        print(f"{model.name:26}{cells}")
    cadence = config.depth.cadence
    per_fused = args.cameras / cadence
    for key, model in (("da2", da2), ("da3", da3)):
        added = speed[key]["total"]["mean"] * per_fused
        print(f"  {model.name}: ~{added:.1f} ms of depth per fused frame "
              f"({args.cameras} cameras, cadence {cadence})")

    print(f"\nWrote {args.out_dir}/summary.json and {len(frames)} comparison images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
