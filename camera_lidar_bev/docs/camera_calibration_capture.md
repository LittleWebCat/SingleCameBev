# Capture front-camera calibration images

This step collects checkerboard images. Camera intrinsics and distortion will be fitted and checked in the next step, after image collection is complete.

The target has **6 × 5 total squares**, each **65 × 65 mm**, giving **5 × 4 inner corners (20 corners)**. Its dimensions are saved in [checkerboard.yaml](../config/calibration/checkerboard.yaml). Keep the checkerboard flat and rigid.

## Start capture

Keep the [front-camera bridge](camera_bridge.md) running in ROS domain 1. In another terminal:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
export DISPLAY=:1
/usr/bin/python3 tools/capture_calibration.py \
  --output data/calibration/front_camera_001 \
  --board-config config/calibration/checkerboard.yaml \
  --target-images 25
```

The default source is `/camera/front/image_raw`. The output directory must be new; the tool refuses an existing directory to avoid mixing sessions. If a previous capture exists, choose a new name such as `front_camera_002`.

The window shows detected corners, their pixel span, and the number of saved images. Press **Space** or **S** to capture a view, then move the board before the next capture. Press **Q** or **Esc** to close the window. A capture requires a fresh image with all 20 corners detected. Very similar poses are rejected; repeatedly pressing Space with a stationary board does not collect useful additional views.

For the next 640 × 480 session, use a new directory such as `front_camera_002` and add `--min-board-span-px 140`. This rejects boards whose larger inner-corner span is below 140 pixels and displays a move-closer message. The chosen size is a collection guide; it does not validate calibration. Keep the whole printed board visible and flat.

## Move the board through varied views

Aim for 25 accepted images distributed across these positions:

1. Start near the image center with the whole board visible.
2. Move toward the left, right, top, and bottom, including near the image edges. Keep all corners and preferably the entire board inside the image.
3. Move closer and farther away to vary the board's apparent size while keeping it fully visible.
4. Tilt the board left, right, upward, and downward. Include different tilts at similar image positions as well as different positions.

Pause and hold the board steady before each capture to reduce motion blur. Avoid glare that hides square boundaries. If corners disappear, reduce the tilt or reposition the board until all 20 are detected again.

Use the camera's actual operating resolution throughout. The verified capture was **640 × 480**; a calibration measured at that size applies to that size. Keep the lens, camera mounting, and exposure settings unchanged during the session. A later resolution or lens/focus change requires checking calibration again.

## Saved evidence and next step

The output directory contains unannotated PNG source images and `session.json`. The manifest preserves detected corner coordinates, image dimensions, ROS image headers, checkerboard dimensions, the user-provided 65 mm square size, and capture evidence from the raw image source. It also records a blur metric; that metric has no universal pass/fail threshold.

Reaching 25 saved images establishes a collection count, not calibration quality. Review the views for coverage, varied tilts, sharpness, and glare before fitting. After collection, report that the images are done. Follow [fitting and review](camera_calibration_fit.md) for the offline helper and held-out checks. The first dataset was rejected and preserved. The second 25-view dataset produced a numerically usable candidate; [the second review](../evaluation/front_camera_002_review.md) describes the remaining independent camera checks. Both capture directories already exist; a future capture requires a new name.
