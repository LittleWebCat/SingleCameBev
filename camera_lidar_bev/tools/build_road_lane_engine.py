#!/usr/bin/env python3
"""Build the pinned official TwinLiteNet BDD100K checkpoint for local TensorRT.

Uses the project's CUDA virtualenv; installs and downloads nothing. The official
640 x 360 RGB / 255 preprocessing and two raw segmentation heads are preserved.
One image inference records functionality, not accuracy or throughput validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

UPSTREAM_COMMIT = "524141821c373c886443cf4254e3e87787c2c55e"
WEIGHTS_SHA256 = "98ffc6c8abc98fda72052117249ce4e2dbc85e58cfcfc6ee2d42f5398c772463"
SOURCE_SHA256 = "dd0f4e7f390f6468d692596543a60cdb219eb26924fb956df8a91ff716268c27"
SHAPES = {
    "image": (1, 3, 360, 640),
    "drivable_logits": (1, 2, 360, 640),
    "lane_logits": (1, 2, 360, 640),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=project / "models/road_lanes_001/best.pth")
    parser.add_argument("--output-directory", type=Path,
                        default=project / "models/road_lanes_001")
    parser.add_argument("--reference-image", type=Path, required=True,
                        help="one existing image used for a functional inference")
    args = parser.parse_args()
    source = project / "third_party/TwinLiteNet/model/TwinLite.py"
    if not args.checkpoint.is_file() or not args.reference_image.is_file():
        parser.error("checkpoint and reference-image must exist locally")
    if sha256(args.checkpoint) != WEIGHTS_SHA256 or sha256(source) != SOURCE_SHA256:
        parser.error("checkpoint or source differs from the pinned official version")
    destination = args.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    onnx_path = destination / "twinlitenet.onnx"
    engine_path = destination / "twinlitenet_fp16.engine"
    provenance_path = destination / "provenance.json"
    if any(p.exists() for p in (onnx_path, engine_path, provenance_path)):
        parser.error("output artifacts already exist; choose a new output-directory")

    import cv2
    import numpy as np
    import onnx
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run with the project's .venv/bin/python")
    torch.set_num_threads(8)
    image = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("reference image could not be decoded")
    resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
    prepared = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None],
                                    dtype=np.float32) / np.float32(255.0)
    spec = importlib.util.spec_from_file_location("official_twinlitenet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TwinLiteNet().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value
                           for key, value in state.items()}, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(prepared), str(onnx_path),
                          input_names=["image"],
                          output_names=["drivable_logits", "lane_logits"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    # A parseable export is needed by the runtime, without an accuracy test suite.
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    print(json.dumps({"stage": "onnx_exported", "parameters": parameter_count,
                      "nodes": len(exported.graph.node)}), flush=True)
    del model, state, exported

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i))
                                     for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.FP16)
    # Keeping optimization modest shortens this one-time device-specific build.
    config.builder_optimization_level = 2
    started = time.perf_counter()
    print(json.dumps({"stage": "building_engine", "layers": network.num_layers}),
          flush=True)
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context() if engine else None
    if context is None:
        raise RuntimeError("fresh TensorRT engine could not be opened")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if set(names) != set(SHAPES):
        raise RuntimeError(f"unexpected engine tensor names: {names}")
    bindings = []
    tensors = {"image": torch.from_numpy(prepared).cuda()}
    for name in names:
        shape = tuple(engine.get_tensor_shape(name))
        if shape != SHAPES[name] or engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f"unexpected engine interface: {name} {shape}")
        if name != "image":
            tensors[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        if not context.set_tensor_address(name, tensors[name].data_ptr()):
            raise RuntimeError(f"could not bind {name}")
        bindings.append({"name": name, "shape": list(shape), "dtype": "FLOAT",
                         "mode": str(engine.get_tensor_mode(name))})
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    # One warmup and one functional image; this is not a benchmark sweep.
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup enqueue failed")
    torch.cuda.synchronize()
    before = torch.cuda.Event(enable_timing=True)
    after = torch.cuda.Event(enable_timing=True)
    before.record(stream)
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference enqueue failed")
    after.record(stream)
    after.synchronize()
    gpu_ms = float(before.elapsed_time(after))
    heads = [tensors[name].cpu().numpy()
             for name in ("drivable_logits", "lane_logits")]
    if any(not np.isfinite(head).all() for head in heads):
        raise RuntimeError("functional inference produced non-finite logits")
    masks = [np.argmax(head[0], axis=0).astype(np.uint8) * 255 for head in heads]
    for name, mask in zip(("drivable", "lane"), masks):
        if not cv2.imwrite(str(destination / f"reference_{name}_mask.png"), mask):
            raise RuntimeError("could not save functional segmentation mask")
    overlay = resized.copy()
    road = masks[0] != 0
    overlay[road] = (overlay[road].astype(np.float32) * 0.68 +
                     np.array([40, 180, 40], np.float32) * 0.32).astype(np.uint8)
    overlay[masks[1] != 0] = (0, 225, 255)
    cv2.imwrite(str(destination / "reference_overlay.png"), overlay)
    provenance = {
        "model": "TwinLiteNet", "task": "drivable area and lane segmentation",
        "training_dataset": "BDD100K", "parameters": parameter_count,
        "upstream_repository": "https://github.com/chequanghuy/TwinLiteNet",
        "upstream_commit": UPSTREAM_COMMIT, "license": "MIT",
        "weights_url": "https://raw.githubusercontent.com/chequanghuy/TwinLiteNet/"
                       f"{UPSTREAM_COMMIT}/pretrained/best.pth",
        "weights_sha256": WEIGHTS_SHA256, "source_sha256": SOURCE_SHA256,
        "onnx_sha256": sha256(onnx_path), "engine_sha256": sha256(engine_path),
        "engine": engine_path.name, "bindings": bindings,
        "preprocessing": {"resize": [640, 360], "resize_mode": "direct stretch",
                          "interpolation": "bilinear", "color": "RGB",
                          "layout": "NCHW", "scale": "divide by 255",
                          "mean_std_normalization": False,
                          "undistortion": False},
        "outputs": {"drivable_logits": "channel 0 background, channel 1 drivable",
                    "lane_logits": "channel 0 background, channel 1 lane marking",
                    "decoding": "argmax channel then nearest resize to native image"},
        "tensorrt_version": trt.__version__, "torch_version": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(0), "internal_precision": "FP16",
        "engine_build_seconds": build_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "functional_inference": {
            "image": str(args.reference_image.resolve()),
            "native_image_size": [image.shape[1], image.shape[0]],
            "gpu_ms_single_sample": gpu_ms,
            "drivable_foreground_pixels": int(np.count_nonzero(masks[0])),
            "lane_foreground_pixels": int(np.count_nonzero(masks[1])),
            "accuracy_validation_performed": False,
            "benchmark_sweep_performed": False,
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"stage": "ready", "engine": str(engine_path),
                      "build_seconds": build_seconds,
                      "functional_inference": provenance["functional_inference"]}),
          flush=True)


if __name__ == "__main__":
    main()
