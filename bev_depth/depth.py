from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import DepthConfig
from .types import Detection


class DepthBackend(Protocol):
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        ...


class ConstantDepthBackend:
    def __init__(self, depth_m: float) -> None:
        self.depth_m = float(depth_m)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        return np.full(frame_bgr.shape[:2], self.depth_m, dtype=np.float32)


class OnnxDepthBackend:
    def __init__(self, model_path: str, input_size: int) -> None:
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Depth ONNX model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = int(input_size)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except cv2.error:
            pass

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self.net.setInput(_depth_preprocess(frame_bgr, self.input_size))
        out = self.net.forward()
        depth = np.squeeze(out).astype(np.float32)
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)


class DepthAnythingV2Backend:
    def __init__(self, model_path: str, input_size: int) -> None:
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Depth Anything checkpoint not found: {model_path}")
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything V2 backend requires torch and the Depth-Anything-V2 "
                "repository on PYTHONPATH. Use backend=tensorrt/onnx on Thor for speed."
            ) from exc

        self.torch = torch
        self.input_size = int(input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        self.model = DepthAnythingV2(**config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            depth = self.model.infer_image(frame_bgr, self.input_size)
        return depth.astype(np.float32)


class MidasTorchBackend:
    def __init__(self, model_name: str, input_size: int) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("MiDaS backend requires torch and Pillow.") from exc

        self.torch = torch
        self.functional = F
        self.image_cls = Image
        self.input_size = int(input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hub_model = model_name or "MiDaS_small"
        self.model = torch.hub.load("intel-isl/MiDaS", hub_model, trust_repo=True)
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]
        image = self.image_cls.fromarray(frame_rgb).resize((self.input_size, self.input_size))
        tensor = self.torch.frombuffer(bytearray(image.tobytes()), dtype=self.torch.uint8)
        tensor = tensor.view(self.input_size, self.input_size, 3).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        with self.torch.inference_mode():
            prediction = self.model(tensor)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = self.functional.interpolate(
                prediction,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
        return prediction.squeeze().detach().cpu().numpy().astype(np.float32)


class TensorRTDepthBackend:
    def __init__(self, model_path: str, input_size: int) -> None:
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Depth TensorRT engine not found: {model_path}")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT depth backend requires NVIDIA TensorRT Python bindings. "
                "Run this on Jetson Thor / JetPack with a pre-built depth engine."
            ) from exc
        try:
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT depth backend requires cuda-python bindings: import cuda.cudart failed."
            ) from exc

        self.trt = trt
        self.cudart = cudart
        self.input_size = int(input_size)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(model_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if -1 in self.input_shape:
            self.input_shape = (1, 3, self.input_size, self.input_size)
            self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.output_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        self.device_buffers: dict[str, int] = {}
        self.host_output = np.empty(int(np.prod(self.output_shape)), dtype=self.output_dtype)
        self._allocate()

    def _allocate(self) -> None:
        for name in self.names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, ptr = self.cudart.cudaMalloc(nbytes)
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
            self.device_buffers[name] = ptr
            self.context.set_tensor_address(name, ptr)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        tensor = _depth_preprocess(frame_bgr, self.input_size)
        input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        tensor = np.ascontiguousarray(tensor.astype(input_dtype, copy=False))
        self.cudart.cudaMemcpyAsync(
            self.device_buffers[self.input_name],
            tensor.ctypes.data,
            tensor.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT depth execution failed")
        self.cudart.cudaMemcpyAsync(
            self.host_output.ctypes.data,
            self.device_buffers[self.output_name],
            self.host_output.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        self.cudart.cudaStreamSynchronize(self.stream)
        depth = self.host_output.reshape(self.output_shape).squeeze().astype(np.float32)
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)


def _depth_preprocess(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    image = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return np.transpose(image, (2, 0, 1))[None, ...]


def create_depth_backend(config: DepthConfig) -> DepthBackend:
    backend = config.backend.lower()
    if backend == "constant":
        return ConstantDepthBackend(config.constant_depth_m or 20.0)
    if backend == "onnx":
        return OnnxDepthBackend(config.path, config.input_size)
    if backend == "depth_anything_v2":
        return DepthAnythingV2Backend(config.path, config.input_size)
    if backend == "midas":
        return MidasTorchBackend(config.path, config.input_size)
    if backend == "tensorrt":
        return TensorRTDepthBackend(config.path, config.input_size)
    raise ValueError(f"Unsupported depth backend: {config.backend}")


def calibrate_depth(raw_depth: float, config: DepthConfig) -> float | None:
    if not np.isfinite(raw_depth) or raw_depth <= 0.0:
        return None
    value = (1.0 / max(raw_depth, 1e-6)) if config.inverse_depth else raw_depth
    metric = value * config.metric_scale + config.metric_bias
    if not np.isfinite(metric) or metric <= 0.0:
        return None
    return float(metric)


def sample_detection_depth(
    depth_map: np.ndarray, detection: Detection, config: DepthConfig
) -> tuple[float | None, str]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    h, w = depth_map.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx1 = int(np.clip(x1 + 0.35 * bw, 0, w - 1))
    cx2 = int(np.clip(x1 + 0.65 * bw, 0, w))
    cy1 = int(np.clip(y1 + 0.65 * bh, 0, h - 1))
    cy2 = int(np.clip(y1 + 0.9 * bh, 0, h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, "invalid_roi"
    roi = depth_map[cy1:cy2, cx1:cx2].astype(np.float32)
    valid = roi[(roi > 0.0) & (roi < 1.0e6)]
    if valid.size == 0:
        return None, "no_valid_depth"
    raw = float(np.median(valid))
    metric = calibrate_depth(raw, config)
    if metric is None:
        return None, "invalid_metric_depth"
    return metric, "depth_map"
