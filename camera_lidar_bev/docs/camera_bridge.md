# Front-camera ROS bridge

The C++ `bev_sensor_bridge/camera_bridge` node captures the selected V4L2 camera through OpenCV and publishes images for recording and input validation. Run it after [building the project](first_steps.md):

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch bev_bringup front_camera.launch.py
```

The node is `/camera_lidar_bev/front_camera_bridge`. Its outputs are:

| Topic | Contents |
| --- | --- |
| `/camera/front/image_raw` | `sensor_msgs/msg/Image`, `bgr8`, actual captured dimensions |
| `/camera/front/camera_info` | `sensor_msgs/msg/CameraInfo`, same stamp and frame ID as the image |
| `/camera_lidar_bev/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray`, status named `camera_lidar_bev/front_camera_bridge` |

## Device and capture settings

The launch defaults select the parent's front-mid camera port. Confirm the physical camera and mounting before treating it as the front sensor.

| Launch argument | Default | Purpose |
| --- | --- | --- |
| `device` | `/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.2:1.0-video-index0` | V4L2 device path |
| `width` | `640` | Requested image width |
| `height` | `480` | Requested image height |
| `fps` | `30.0` | Requested capture rate |
| `fourcc` | `MJPG` | Requested camera format |
| `params_file` | Installed `config/sensors/front_camera.yaml` | Frame, topic, bounded-run, and freshness configuration |
| `calibration_file` | Empty | Path to measured ROS camera-calibration YAML |
| `source_mode` | `camera` | `camera` for V4L2, `synthetic` for software validation |
| `use_sim_time` | `false` | Must remain false for both capture and synthetic modes |

For example:

```bash
ros2 launch bev_bringup front_camera.launch.py \
  device:=/dev/video0 width:=640 height:=480 fps:=30.0 fourcc:=MJPG
```

The bridge requests a four-frame capture buffer. Driver support and capture timing must be measured. A negotiated resolution different from the requested width/height is an error diagnostic; messages still describe the actual captured dimensions. The bridge does not silently choose a different camera.

Capture launch arguments override corresponding entries in `params_file`; use those arguments to choose the device, dimensions, FPS, codec, source mode, and calibration file. The default parameter file configures `frame_id` (`camera_front_optical_frame`), the image and camera-information topics above, `max_frames` (`0`, unlimited), and `stale_after_s` (`1.0`). A positive `max_frames` ends the node after that many published frames.

If opening the device fails, inspect the path, device access, and any process already capturing from it. Some V4L2 drivers allow only one capture owner. During read failures, no new images are published; the timer continues attempting reads and diagnostics report the failure. Diagnostics can report stale reception while a camera read is blocked. Stopping the node can still wait for the V4L2 backend read timeout; this capture path does not guarantee a bounded shutdown time.

## Timestamp contract

Each image receives the node's ROS time immediately after a successful read returns. Its camera information receives the identical header. This stamp includes camera, transport, buffering, and decoding delay; it does not identify the hardware exposure time.

The bridge reports unverified sensor timing as a warning even when calibration is loaded; capture or calibration errors raise the status to error. A recent message header establishes host reception freshness only. Camera–LiDAR time alignment requires independent measurement. The bridge rejects `use_sim_time:=true` in both camera and synthetic modes because its stamps use host reception time. For bag replay, start only the input probe with simulated time and replay the recorded image and camera-information topics.

The image frame ID identifies the camera optical frame. No camera–`base_link` TF or sensor extrinsics are inferred by this bridge.

## Camera calibration

The current session uses the [imported native 1024 × 768 camera YAML](imported_calibration.md). Its 58 paired live image/CameraInfo messages were verified. It remains provisional, and the imported extrinsic transform's source frame is still unconfirmed. Use the explicit dimensions and file path from that review for a restart.

Without calibration, `CameraInfo.k[0]` is zero, marking the camera uncalibrated. Images are still available for recording. Supply a standard ROS camera-calibration YAML measured at the operating dimensions:

```bash
ros2 launch bev_bringup front_camera.launch.py \
  calibration_file:=/absolute/path/to/front_camera_calibration.yaml
```

The YAML requires `image_width`, `image_height`, `camera_matrix`, `distortion_model`, `distortion_coefficients`, `rectification_matrix`, and `projection_matrix`; `camera_name` is optional. Matrix fields use `rows`, `cols`, and `data`. The loader supports `plumb_bob` with five distortion coefficients, `rational_polynomial` with eight, and `equidistant` with four. It checks matrix dimensions and finite, valid calibration values. Store actual measured coefficients. The project does not provide invented intrinsics.

A missing or invalid calibration file prevents startup. If valid calibration dimensions differ from the captured images, publication continues with uncalibrated camera information and an error diagnostic. The bridge does not rescale intrinsics. Recalibrate at the operating dimensions or select the dimensions used for the calibration.

[calibration.example.yaml](../config/calibration/calibration.example.yaml) describes future multi-sensor calibration; it is not the camera YAML accepted here. Intrinsics alone do not establish sensor transforms or timing alignment.

The earlier second checkerboard dataset has a saved [640 × 480 candidate YAML](../config/calibration/front_camera_640x480_002.candidate.yaml). Following the user's straight-edge check, it was applied provisionally and [59 paired headers were verified](../evaluation/front_camera_002_application.json). That record describes the earlier session. The current imported camera candidate and native 1024 × 768 dimensions are documented above. Intrinsic accuracy remains unvalidated; the diagnostic's `calibrated: true` reports file loading at matching dimensions.

## Validation

To open a direct camera window while the bridge is running:

```bash
source /opt/ros/jazzy/setup.bash
DISPLAY=:1 /usr/bin/python3 tools/view_camera.py
```

The viewer subscribes to `/camera/front/image_raw` with sensor-data Best Effort QoS. Press `q` or `Esc` to close it. It marks stale input in the window. To inspect one received image without opening a window, use `--snapshot /tmp/front_camera.png`.

After building and sourcing the overlay, run the hardware-independent integration check:

```bash
/usr/bin/python3 tools/smoke_test_camera_bridge.py
```

It uses synthetic frames to check ROS publication, matching image/camera-information headers, calibration behavior, and error handling. Passing this check does not establish camera throughput, hardware timestamp quality, or sensor alignment.

For live capture, inspect actual messages and reception rate in another sourced terminal:

```bash
ros2 topic hz /camera/front/image_raw
ros2 topic echo /camera/front/camera_info --once
ros2 topic echo /camera_lidar_bev/diagnostics
```

The input probe may publish on the same diagnostic topic; identify the camera status by its name. Expect a timing warning even when images arrive successfully. Complete [the first recording report](../evaluation/first_recording.md) with measured dimensions, rates, gaps, timestamp sources, and calibration status before moving on to camera–LiDAR fusion.

The initial [validation report](../evaluation/camera_bridge_smoke.md) records software checks and a short physical-camera run. It distinguishes the bridge's publication rate from delivered subscriber messages; longer recording and timing measurements are still required.
