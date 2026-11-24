#!/usr/bin/env python3
"""Measure fixed-batch VisionPilot TensorRT engines on this machine.

This is a model-speed benchmark, not the VisionPilot application: AutoDrive
receives unwarped road images because this camera has no VisionPilot homography.
Run with GPU access, for example:

  .venv/bin/python scripts/benchmark_visionpilot.py --engine-dir runs/engines \
      --video video.mkv --output runs/visionpilot_benchmark.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

SHAPE = (1, 3, 512, 1024)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
MODELS = ("autodrive", "autosteer", "autospeed")


def checked(result, operation: str):
    error, *values = result
    if error != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation}: {error}")
    return values[0] if len(values) == 1 else tuple(values)


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    return {
        "mean": mean, "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
        "min": float(a.min()), "max": float(a.max()),
        "std": float(a.std()), "samples": len(values),
    }


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip("\x00\n ")
    except OSError:
        return None


def command_output(argv: list[str]) -> str | None:
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        return run.stdout.strip() if run.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def hardware_metadata() -> dict:
    device = checked(cuda.cudaGetDeviceProperties(0), "cudaGetDeviceProperties")
    device_name = device.name
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace").rstrip("\0")
    free, total = checked(cuda.cudaMemGetInfo(), "cudaMemGetInfo")
    thermal = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        temperature = read_text(str(zone / "temp"))
        if temperature:
            thermal[read_text(str(zone / "type")) or zone.name] = temperature
    frequencies = {}
    for path in sorted(Path("/sys/class/devfreq").glob("*/cur_freq")):
        frequencies[path.parent.name] = read_text(str(path))
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
        "tensorrt": trt.__version__,
        "cuda_runtime_version": checked(cuda.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion"),
        "cuda_driver_version": checked(cuda.cudaDriverGetVersion(), "cudaDriverGetVersion"),
        "gpu": {"name": str(device_name), "compute_capability": f"{device.major}.{device.minor}",
                "multiprocessors": int(device.multiProcessorCount),
                "global_memory_bytes": int(device.totalGlobalMem),
                "free_memory_bytes": int(free), "total_memory_bytes": int(total)},
        "device_tree_model": read_text("/proc/device-tree/model"),
        "jetson_release": read_text("/etc/nv_tegra_release"),
        "power_mode": command_output(["nvpmodel", "-q"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,power.draw,power.limit,"
                                      "clocks.sm,temperature.gpu", "--format=csv,noheader"]),
        "thermal_millidegrees_c": thermal, "devfreq_hz": frequencies,
    }


class Runner:
    """An independent TensorRT context, stream, and pinned IO buffers."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.device: dict[str, int] = {}
        self.host_ptr: dict[str, int] = {}
        self.host: dict[str, np.ndarray] = {}
        self.events = []
        self.stream = None
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create TensorRT context for {path}")
        try:
            self.stream = checked(cuda.cudaStreamCreateWithFlags(1), "cudaStreamCreateWithFlags")
            for index in range(self.engine.num_io_tensors):
                tensor = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                    self.inputs.append(tensor)
                    if not self.context.set_input_shape(tensor, SHAPE):
                        raise RuntimeError(f"Cannot set {tensor} shape to {SHAPE}")
                else:
                    self.outputs.append(tensor)
            expected_inputs = {"image_prev", "image_curr"} if name == "autodrive" else {"input"}
            if set(self.inputs) != expected_inputs:
                raise RuntimeError(f"Unexpected {name} inputs: {self.inputs}; expected {expected_inputs}")
            self.tensor_metadata = {}
            for tensor in self.inputs + self.outputs:
                shape = tuple(self.context.get_tensor_shape(tensor))
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor)))
                if not shape or any(dim <= 0 for dim in shape):
                    raise RuntimeError(f"Unresolved tensor shape {tensor}: {shape}")
                if tensor in self.inputs and (shape != SHAPE or dtype != np.dtype(np.float32)):
                    raise RuntimeError(f"Expected float32 {SHAPE}, got {tensor}: {dtype} {shape}")
                nbytes = int(np.prod(shape)) * dtype.itemsize
                ptr = checked(cuda.cudaMalloc(nbytes), f"cudaMalloc({tensor})")
                self.device[tensor] = ptr
                pinned = checked(cuda.cudaHostAlloc(nbytes, 0), f"cudaHostAlloc({tensor})")
                self.host_ptr[tensor] = pinned
                buffer = (ctypes.c_ubyte * nbytes).from_address(int(pinned))
                self.host[tensor] = np.ndarray(shape, dtype=dtype, buffer=buffer)
                if not self.context.set_tensor_address(tensor, int(ptr)):
                    raise RuntimeError(f"Cannot bind tensor {tensor}")
                self.tensor_metadata[tensor] = {
                    "mode": "input" if tensor in self.inputs else "output",
                    "shape": list(shape), "dtype": str(dtype), "bytes": nbytes,
                }
            for _ in range(3):
                self.events.append(checked(cuda.cudaEventCreate(), "cudaEventCreate"))
            self.inference_start, self.inference_end, self.complete = self.events
        except BaseException:
            self.close()
            raise

    def stage(self, tensors: dict[str, np.ndarray]) -> None:
        for name in self.inputs:
            np.copyto(self.host[name], tensors[name], casting="no")

    def upload(self) -> None:
        for name in self.inputs:
            checked(cuda.cudaMemcpyAsync(self.device[name], self.host_ptr[name], self.host[name].nbytes,
                                         cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream),
                    f"H2D {self.name}/{name}")

    def execute(self) -> None:
        checked(cuda.cudaEventRecord(self.inference_start, self.stream), "record inference start")
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError(f"TensorRT execution failed: {self.name}")
        checked(cuda.cudaEventRecord(self.inference_end, self.stream), "record inference end")

    def download(self) -> None:
        for name in self.outputs:
            checked(cuda.cudaMemcpyAsync(self.host_ptr[name], self.device[name], self.host[name].nbytes,
                                         cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream),
                    f"D2H {self.name}/{name}")

    def synchronize(self) -> None:
        checked(cuda.cudaStreamSynchronize(self.stream), f"synchronize {self.name}")

    def validate(self) -> None:
        for name in self.outputs:
            if not np.isfinite(self.host[name]).all():
                raise RuntimeError(f"Nonfinite output: {self.name}/{name}")

    def metadata(self) -> dict:
        output_stats = {}
        for name in self.outputs:
            a = self.host[name]
            output_stats[name] = {"all_finite": bool(np.isfinite(a).all()),
                                  "min": float(a.min()), "max": float(a.max()),
                                  "mean": float(a.mean(dtype=np.float64))}
        return {"engine": str(self.path.resolve()), "engine_bytes": self.path.stat().st_size,
                "engine_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "io_tensors": self.tensor_metadata, "last_output_statistics": output_stats}

    def close(self) -> None:
        # Cleanup is also used on partial initialization. Preserve the original
        # failure while making any CUDA cleanup errors visible on stderr.
        actions = []
        if self.stream is not None:
            actions.append((cuda.cudaStreamSynchronize, self.stream))
        actions.extend((cuda.cudaEventDestroy, event) for event in self.events)
        actions.extend((cuda.cudaFree, ptr) for ptr in self.device.values())
        actions.extend((cuda.cudaFreeHost, ptr) for ptr in self.host_ptr.values())
        if self.stream is not None:
            actions.append((cuda.cudaStreamDestroy, self.stream))
        for function, resource in actions:
            try:
                checked(function(resource), function.__name__)
            except Exception as exc:
                print(f"CUDA cleanup: {exc}", file=sys.stderr)
        self.events.clear()
        self.device.clear()
        self.host_ptr.clear()
        self.host.clear()
        self.stream = None


def load_frames(args) -> tuple[list[np.ndarray], dict]:
    if args.video is None:
        rng = np.random.default_rng(20261015)
        frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
                  for _ in range(args.frames)]
        return frames, {"type": "synthetic", "seed": 20261015, "frames": len(frames),
                        "warning": "Synthetic images test speed only; prefer --video road footage."}
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {args.video}")
    frames = []
    frame_indices = []
    metadata = {"type": "video", "path": str(args.video.resolve()),
                "file_bytes": args.video.stat().st_size,
                "fps": capture.get(cv2.CAP_PROP_FPS),
                "total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    try:
        for index in range(args.frames * args.frame_stride):
            ok, frame = capture.read()
            if not ok:
                break
            if index % args.frame_stride == 0:
                frames.append(frame)
                frame_indices.append(index)
                if len(frames) == args.frames:
                    break
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError("Need at least two decoded video frames for AutoDrive")
    metadata.update({"frames": len(frames), "frame_indices": frame_indices,
                     "frame_shape_hwc": list(frames[0].shape)})
    return frames, metadata


def preprocess(frame: np.ndarray) -> np.ndarray:
    # Official top-crop means REMOVE top pixels, preserving the lower 2:1 image.
    height, width = frame.shape[:2]
    crop_top = max(0, height - width // 2)
    resized = cv2.resize(frame[crop_top:, :], (1024, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def prepare_pair(previous: np.ndarray, current: np.ndarray) -> dict:
    prev = preprocess(previous)
    curr = preprocess(current)
    return {"autodrive": {"image_prev": (prev - MEAN) / STD,
                           "image_curr": (curr - MEAN) / STD},
            "autosteer": {"input": curr}, "autospeed": {"input": curr}}


class GroupTimer:
    def __init__(self):
        self.stream = checked(cuda.cudaStreamCreateWithFlags(1), "create timing stream")
        self.start = checked(cuda.cudaEventCreate(), "create timing start")
        self.end = checked(cuda.cudaEventCreate(), "create timing end")

    def run(self, runners: list[Runner], concurrent: bool, transfers: bool) -> tuple:
        if not transfers:
            for runner in runners:
                runner.upload()
            for runner in runners:
                runner.synchronize()
        begin = time.perf_counter_ns()
        checked(cuda.cudaEventRecord(self.start, self.stream), "record group start")
        previous_event = None
        for runner in runners:
            checked(cuda.cudaStreamWaitEvent(runner.stream, self.start, 0), "wait group start")
            if not concurrent and previous_event is not None:
                checked(cuda.cudaStreamWaitEvent(runner.stream, previous_event, 0), "serialize models")
            if transfers:
                runner.upload()
            runner.execute()
            if transfers:
                runner.download()
            checked(cuda.cudaEventRecord(runner.complete, runner.stream), "record model completion")
            previous_event = runner.complete
        for runner in runners:
            checked(cuda.cudaStreamWaitEvent(self.stream, runner.complete, 0), "wait model completion")
        checked(cuda.cudaEventRecord(self.end, self.stream), "record group end")
        checked(cuda.cudaEventSynchronize(self.end), "synchronize group end")
        wall_ms = (time.perf_counter_ns() - begin) / 1e6
        gpu_ms = float(checked(cuda.cudaEventElapsedTime(self.start, self.end), "elapsed group time"))
        per_model = {runner.name: float(checked(cuda.cudaEventElapsedTime(
            runner.inference_start, runner.inference_end), "elapsed inference")) for runner in runners}
        if not transfers:
            # Output copies occur after all model timing has finished, so their
            # bandwidth cannot contend with another model's measured execution.
            for runner in runners:
                runner.download()
            for runner in runners:
                runner.synchronize()
        return wall_ms, gpu_ms, per_model

    def close(self) -> None:
        checked(cuda.cudaEventDestroy(self.start), "destroy timing start")
        checked(cuda.cudaEventDestroy(self.end), "destroy timing end")
        checked(cuda.cudaStreamDestroy(self.stream), "destroy timing stream")


def benchmark(args, timer, runners, samples, frames, *, concurrent, transfers, with_preprocess=False):
    wall_values, gpu_values, preprocessing_values = [], [], []
    per_model = {runner.name: [] for runner in runners}
    for index in range(args.warmup + args.iterations):
        sample_index = index % len(samples)
        prepared_begin = time.perf_counter_ns()
        if with_preprocess:
            current = sample_index + 1
            sample = prepare_pair(frames[current - 1], frames[current])
        else:
            sample = samples[sample_index]
        for runner in runners:
            runner.stage(sample[runner.name])
        staging_ms = (time.perf_counter_ns() - prepared_begin) / 1e6
        wall_ms, gpu_ms, model_ms = timer.run(runners, concurrent, transfers)
        # This check is outside model and prepared-frame latency measurements.
        for runner in runners:
            runner.validate()
        if index >= args.warmup:
            wall_values.append(wall_ms + staging_ms if with_preprocess else wall_ms)
            gpu_values.append(gpu_ms)
            preprocessing_values.append(staging_ms)
            for name, value in model_ms.items():
                per_model[name].append(value)
    result = {
        "wall_latency_ms": summary(wall_values), "gpu_group_span_ms": summary(gpu_values),
        "per_model_gpu_execution_ms": {name: summary(values) for name, values in per_model.items()},
        "frames_per_second_from_mean_wall_latency": 1000.0 / float(np.mean(wall_values)),
        "frames_per_second_from_mean_gpu_span": 1000.0 / float(np.mean(gpu_values)),
        "transfers_inside_timer": transfers, "concurrent_streams": concurrent,
        "all_outputs_finite": True,
        "validated_samples_per_model": args.warmup + args.iterations,
    }
    if with_preprocess:
        result["cpu_preprocessing_and_pinned_staging_ms"] = summary(preprocessing_values)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-preprocessing", action="store_true")
    args = parser.parse_args()
    if args.frames < 2 or args.frame_stride < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("Need frames>=2, frame-stride>=1, warmup>=0, iterations>=1")
    return args


def main() -> None:
    args = parse_args()
    frames, source = load_frames(args)
    samples = [prepare_pair(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    report = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_commit": args.source_commit, "precision_requested": args.precision,
        "batch_size": 1, "input_shape_nchw": list(SHAPE),
        "warmup_per_measurement": args.warmup, "iterations_per_measurement": args.iterations,
        "source": source, "hardware_before": hardware_metadata(), "measurements": {},
        "scope": {
            "description": "Unwarped image model-speed benchmark; not full VisionPilot application FPS.",
            "preprocessing": "Remove max(0,h-w//2) top rows; OpenCV linear resize 1024x512; BGR->RGB; "
                             "float32 /255. AutoDrive additionally uses ImageNet mean/std.",
            "temporal_inputs": "AutoDrive image_prev/image_curr use adjacent sampled frames in temporal order.",
            "missing": ["camera homography / AutoDrive BEV warp", "camera capture / video decoding",
                        "fusion", "planning", "rendering", "control"],
            "compute_only": "Inputs uploaded before timing; outputs downloaded and validated after timing.",
            "transfers_and_compute": "Pinned inputs staged before timing; timer includes explicit H2D, "
                                     "TensorRT enqueue/execution, D2H and synchronization.",
            "gpu_group_span": "CUDA events on a control stream spanning all participating streams, "
                              "including submission gaps; excludes output copies in compute-only mode.",
            "model_gpu_execution": "CUDA events directly around TensorRT execute_async_v3 in each model stream.",
            "throughput": "Reciprocal of mean synchronized single-frame latency, with no cross-frame pipelining. "
                          "Each group frame executes all three models, not three independent video frames.",
            "prepared_frame": "CPU crop/resize/color conversion/normalization of a previous/current pair, "
                              "pinned-buffer staging, concurrent H2D/inference/D2H; excludes missing stages above.",
        },
    }
    runners = []
    timer = None
    try:
        for name in MODELS:
            runners.append(Runner(name, args.engine_dir / f"{name}_{args.precision}.engine"))
        timer = GroupTimer()
        for transfers in (False, True):
            category = "transfers_and_compute" if transfers else "compute_only"
            report["measurements"][category] = {}
            groups = [(runner.name, [runner], False) for runner in runners]
            groups += [("all_models_sequential", runners, False), ("all_models_concurrent", runners, True)]
            for name, group, concurrent in groups:
                print(f"Benchmark {category}/{name}: warmup={args.warmup}, iterations={args.iterations}",
                      file=sys.stderr, flush=True)
                result = benchmark(args, timer, group, samples, frames,
                                   concurrent=concurrent, transfers=transfers)
                report["measurements"][category][name] = result
                print(f"  wall {result['wall_latency_ms']['mean']:.3f} ms; "
                      f"GPU span {result['gpu_group_span_ms']['mean']:.3f} ms", file=sys.stderr, flush=True)
        if not args.skip_preprocessing:
            print("Benchmark prepared_frame_concurrent", file=sys.stderr, flush=True)
            report["measurements"]["prepared_frame_concurrent"] = benchmark(
                args, timer, runners, samples, frames, concurrent=True, transfers=True, with_preprocess=True)
        report["models"] = {runner.name: runner.metadata() for runner in runners}
        report["hardware_after"] = hardware_metadata()
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}", flush=True)
    finally:
        if timer is not None:
            timer.close()
        for runner in reversed(runners):
            runner.close()


if __name__ == "__main__":
    main()
