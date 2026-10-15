# Front-camera bridge validation

Scope: initial ROS camera input, not camera–LiDAR fusion or a sustained perception benchmark.

## Build and software checks

- ROS 2 Jazzy, C++17, OpenCV 4.8.0, yaml-cpp 0.8.0.
- All three project packages build with the system ROS Python interpreter.
- CTest: grid tests and camera-support tests pass.
- Camera-support coverage includes absent/invalid calibration, model and matrix validation, resolution mismatch, shared headers, and packed/non-contiguous BGR images.
- Synthetic ROS checks pass for uncalibrated publication, loaded calibration, mismatched calibration, finite capture termination, missing-device failure, and rejection of simulated-time capture.

## Short physical-camera check

| Item | Observed result |
| --- | --- |
| Source | `/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.2:1.0-video-index0` (`/dev/video2`) |
| Requested/backend-reported mode | 640 × 480 MJPG, 30 FPS |
| Published images | 90 |
| Bridge mean receipt rate | 30.02 Hz between first and last published frame |
| Test subscriber received images | 72 |
| Received images with matching CameraInfo header | 72 |
| Subscriber rate over its received sample window | 28.41 Hz |
| Image encoding / step | `bgr8` / 1920 bytes |
| Calibration | Not supplied; received `CameraInfo.k[0]` was zero |
| Timestamp source | Host system time after OpenCV read returns |
| Exposure timing / synchronization | Unverified |

The finite check ran on an isolated localhost ROS domain and saved no images. The subscriber uses sensor-data best-effort QoS with a small queue; discovery and message drops can reduce received counts. This check did not identify which mechanism accounts for every missing sample. Publication rate and subscriber delivery rate are separate results.

Next evidence: camera + LiDAR + validated motion recording/replay, measured timestamp relationships, independent camera calibration, and longer recording-load tests. Neither this short camera run nor synthetic calibration validates the complete 20 Hz BEV target.
