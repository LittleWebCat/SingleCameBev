# Review the imported front-camera and LiDAR calibration

The user supplied `front.json` and `front_external.json` and confirmed that they were calibrated with the same camera and LiDAR in their current mounting positions. The images were captured directly at **1024 × 768**. The source files do not name a LiDAR frame. The user selected **B (`os_sensor`)** in the live A/B comparison; this is the working hypothesis, with accuracy still provisional.

Byte-preserved copies and candidate conversions are in [imported_front_001](../config/calibration/imported_front_001/import_report.json). The imported transform remains unvalidated; it has not been published as ROS TF.

## What the files contain

| File | Content | Reported fitting error |
| --- | --- | --- |
| `front.json` | Pinhole K and five distortion coefficients; 40 checkerboard views at 1024 × 768 | 0.320 pixels RMS |
| `front_external.json` | LiDAR-to-camera R/t in meters, both homogeneous transform directions, and fit diagnostics | 4.869 pixels RMS for 68 inliers; 8.897 pixels for all 120 points |

The board definition matches 5 × 4 inner corners and 65 mm square size. Rotation is orthonormal with determinant +1. The transform matrices agree with R/t and with each other's inverses. The external file's embedded camera parameters exactly match `front.json`; the stored residual summaries recompute correctly.

The external `pair_files` list contains `pose4_pairs.json` twice. Its corresponding 20-point error blocks are exactly duplicated. This weights one pose more than once; the nominal six pair-file entries contain five unique filenames. The original correspondence files are unavailable here, so this import preserves the fitted values and records the issue rather than pretending to refit them. These errors are training residuals, without independent held-out accuracy evidence.

The inverse translation places the camera origin at approximately `[0.45745, 0.01975, -0.23270]` meters in the calibration's claimed LiDAR coordinates, with separation 0.514 m. This is a numerical interpretation of the supplied file, not an independent mounting measurement.

## Native-resolution camera application

The front camera's V4L2 driver supports 1024 × 768 MJPG at nominal 30 fps. The bridge was started at those dimensions with [the converted camera YAML](../config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml). [Application evidence](../evaluation/imported_front_application.json) verifies 58 live image/CameraInfo header pairs and full K/D/R/P agreement. Timing remains WARN because exposure timestamps and sensor synchronization are unverified.

The earlier 640 × 480 candidate and its fitting/application reports are retained as historical evidence. No intrinsics were implicitly scaled to the new dimensions. For a future restart:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=1 ros2 launch bev_bringup front_camera.launch.py \
  width:=1024 height:=768 fps:=30.0 \
  calibration_file:=/home/agxthor/SingleCameraBev/camera_lidar_bev/config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml
```

Start a single Ouster driver using [the existing bring-up procedure](lidar_bringup.md). It requires the powered sensor and physical Ethernet link. The camera application is provisional and does not establish camera–LiDAR alignment.

## Compare the two source-frame possibilities

Ouster's LiDAR and sensor frames have opposite X/Y axes. The driver publishes the actual `os_sensor` parent / `os_lidar` child transform, including a 0.038195 m vertical offset for this sensor. [Ouster's coordinate documentation](https://static.ouster.dev/sensor-docs/image_route1/image_route3/sensor_data/sensor-data.html) explains the frame distinction. The preview reads current driver TF rather than assuming it.

The current `/points` cloud uses `os_lidar`. The comparison keeps both imported interpretations explicit:

- **A:** the supplied transform maps `os_lidar` directly to the camera.
- **B:** the supplied transform maps `os_sensor` to the camera; compose it with the observed `T_os_sensor_from_os_lidar` before projecting the current cloud.

Keep the rig and scene stationary, then open the diagnostic:

```bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/view_imported_projection.py
```

It projects measured points into the **raw, distorted, native-resolution** image using K/D and the stated transform direction. White markers have black outlines for visibility. In comparison mode, the two panes remain labeled **UNCONFIRMED**. That mode does not select a source frame or publish camera–LiDAR TF.

The first live [comparison report](../evaluation/imported_projection_001.json) and [snapshot](../data/calibration/imported_projection_preview_001.png) were saved after restoring the LiDAR Ethernet connection and driver. They contain a 1024 × 768 image paired with a 262,144-point `os_lidar` cloud, with 12.0 ms header separation. The observed driver TF is used for B. All 18 importer/projection checks passed, including the radial fold-back regression. Reproducible raw evidence is in the ignored `data/calibration/imported_projection_001` directory.

The initial scene is largely a plain curved surface, which provides limited independent alignment evidence. The user preferred B. This qualitative choice is recorded separately from the original comparison; the presence of projected points alone is not an accuracy test.

The tool requires matching image/CameraInfo headers and matrices, fresh image/cloud receipts, a cloud in `os_lidar`, and an image/cloud header difference no greater than 150 ms. This is a pairing heuristic, not exposure synchronization. Stopped inputs are visibly marked stale. The distorted projection is limited to the initial monotonic radial branch so out-of-domain rays cannot fold back into the image.

Press **Q** or **Esc** to close and **S** to save a fresh diagnostic. To save one pair headlessly with reproducible raw evidence, use new output paths:

```bash
ROS_DOMAIN_ID=1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/view_imported_projection.py \
  --snapshot /tmp/imported_front_comparison.png \
  --report /tmp/imported_front_comparison.json \
  --evidence-dir data/calibration/imported_projection_001
```

## Working candidate B

[The selection record](../evaluation/imported_frame_selection_B.json) preserves the user's B response and the source/comparison hashes. [The B candidate](../config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml) contains the original transform interpreted as `T_camera_from_os_sensor` and its composition for live `os_lidar` points:

```text
T_camera_from_os_lidar = T_camera_from_os_sensor @ T_os_sensor_from_os_lidar
```

The original UNKNOWN-frame candidate and comparison report are retained. The observed driver transform flips X/Y and adds 0.038195 m to Z before applying the external transform. The composed camera translation is `[0.0129217294, -0.2325992185, -0.4782952600]` m. The preview continues to read actual driver TF.

With the camera and LiDAR already running, open a single B pane:

```bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/view_imported_projection.py --view b
```

The pane labels B as the user's visual choice and accuracy as provisional. This uses B for diagnostic viewing; it does not publish an extrinsic TF or qualify the calibration for fusion.

The first [single-B report](../evaluation/imported_projection_B_001.json) and [snapshot](../data/calibration/imported_projection_B_preview_001.png) contain fresh native 1024 × 768 inputs with 15.6 ms header separation. The live composed matrix matches the saved B candidate. Reproducible raw evidence is in `data/calibration/imported_projection_B_001`. All 12 projection tests passed, including single-pane composition and provisional-report checks; the earlier 8 importer checks remain valid.

First hold the board still in the middle of the camera view, separated from the background. Inspect the projected points along its **outer physical edges**; black/white printed corner lines are not LiDAR geometry. Then check the left and right of the image and at a second distance. Independent projection checks and a unique-correspondence refit remain the next requirements before adopting the extrinsic transform for fusion.
