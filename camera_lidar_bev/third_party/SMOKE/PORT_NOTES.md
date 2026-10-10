# SMOKE inference port

The minimal inference adapter retains all 233 pretrained parameters and exact layer names. Training, optimizer state and the legacy PyTorch CUDA extension are excluded. Torchvision modulated deformable convolution replaces the original CUDA operation with the same offset ordering. ONNX uses NVIDIA TensorRT ModulatedDeformConv2d plugin version 2 with the documented input order and fields. The original group normalization, learned dimension offsets, normalized orientation vector and heatmap sigmoid/clamp are preserved.

Input: FLOAT BGR normalized NCHW [1,3,384,1280]. Outputs: FLOAT heatmap [1,3,96,320], regression [1,8,96,320]. Classes: Car, Cyclist, Pedestrian. Outputs require the original SMOKE decoder and calibrated camera intrinsics. The uniform centered affine crops the top and bottom of tall source images; this is documented in model provenance.

Official pretrained weights are downloaded separately and are kept under ignored model directories. No checkpoint is committed. Numeric and speed measurements are recorded by the builder; physical accuracy remains unvalidated.

The first export was checked against Torch on the calibrated KITTI 000008 demo: seven reference car detections agreed within 0.056 m in position, 0.0036 m in dimension, 0.0021 radians in heading and 0.24 image pixels for projected corners. These numbers establish port agreement and are not ground-truth accuracy. The engine took about 25 ms on AGX Thor, excluding capture, preprocessing, decoding and display; see model provenance for the exact measurement.

The builder safely loads the pinned checkpoint with `weights_only=True` and requires its SHA256 `e2cb8a1643b37b275907867716fce1f682f113dc6dd2dda141ffa007023f9546`. A reference image must contain at least one supported detection before an expensive export is started. Native ROS camera YAML or the official KITTI demo flat 4x4 projection text may be supplied. A future build can use:

```bash
.venv/bin/python camera_lidar_bev/tools/build_camera_3d_engine.py \
  --checkpoint /path/to/model_final.pth \
  --output-directory camera_lidar_bev/models/camera_3d_NEW \
  --reference-image camera_lidar_bev/data/camera_3d_demo_001/kitti_000008.png \
  --camera-calibration camera_lidar_bev/data/camera_3d_demo_001/kitti_000008.calib.txt
```

Monocular learned depth uses the original fixed depth reference; intrinsics unproject the predicted depth but do not remove the training camera's focal-length bias. The live front camera differs from KITTI and uses a tall image with a central crop. Its range and dimensions therefore need separate validation.
