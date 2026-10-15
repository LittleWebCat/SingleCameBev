# Fit and review front-camera intrinsics

Fit saved views at the operating resolution, using the measured board dimensions. This board has 6 × 5 total squares, 5 × 4 inner corners, and 65 mm squares. The [capture instructions](camera_calibration_capture.md) explain how to collect views.

The current camera session uses [imported native 1024 × 768 intrinsics](imported_calibration.md), confirmed by the user for the current rig. The 640 × 480 fitting results below describe the earlier project dataset and remain preserved; select a calibration that matches the actual capture dimensions.

The second 25-view dataset produced a numerically usable [640 × 480 candidate calibration](../config/calibration/front_camera_640x480_002.candidate.yaml): held-out RMS is 0.611 pixels and training subset focal changes are at most 2.09%. The user reported that straight edges remain straight in its corrected preview. This candidate was applied provisionally in the earlier session; independent numerical accuracy remains unvalidated. Follow [the second review](../evaluation/front_camera_002_review.md) for that historical fitting/application evidence. The first-dataset capture and rejection instructions below are retained as evidence.

Run the offline fitting helper with system Python:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
/usr/bin/python3 tools/calibrate_camera.py \
  --images data/calibration/front_camera_001 \
  --columns 5 --rows 4 --square-size-mm 65 \
  --output config/calibration/front_camera_640x480.candidate.yaml \
  --report evaluation/front_camera_001_fit.json
```

The helper re-detects corners in the raw images and checks the capture manifest's board and dimensions. It selects deterministic held-out images before fitting. With 25 usable views and default settings, it fits 20 and holds out 5. The exported camera matrix and distortion come only from the training views.

The report includes source hashes, the exact split, per-view errors, held-out reprojection errors, approximate parameter uncertainties, image coverage, training subset stability, and a radial distortion diagnostic. Held-out evaluation fixes the lens parameters but estimates a board pose for each held-out image. That can absorb lens errors; a low error alone does not establish accurate intrinsics.

The five-coefficient `plumb_bob` radial mapping must remain increasing and positive across the tested normalized pinhole image domain. A folded model is rejected: the helper preserves numerical evidence in JSON, returns exit code 2, and writes no camera YAML. This domain is an extrapolation diagnostic based on the fitted matrix, not a measured field of view. OpenCV also identifies non-monotonic distortion as a calibration failure; see its [calibration documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

A numerically usable fit is saved as `candidate_unvalidated`. Review coverage, varied tilts, parameter stability, and independent image-edge observations before applying it. The tool does not start or reconfigure the camera bridge. Intrinsic calibration does not measure camera–LiDAR transforms or timing.

## First dataset needs another capture

The 25 images in `front_camera_001` are preserved. Most show a small board in a narrow horizontal band; the trial lens model fails the radial diagnostic. See [the review](../evaluation/front_camera_001_review.md) and [the detailed fitting report](../evaluation/front_camera_001_fit.json).

For the next set, keep the entire printed board flat, rigid, visible, and roughly one-third to two-thirds of the image width. At 640 × 480, start with the inner corners spanning at least 140 pixels along the larger image axis. This size is a collection guide, not an accuracy threshold. Move closer instead of walking farther away.

With the [camera bridge](camera_bridge.md) running, open a new capture session:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
DISPLAY=:1 /usr/bin/python3 tools/capture_calibration.py \
  --output data/calibration/front_camera_002 \
  --board-config config/calibration/checkerboard.yaml \
  --target-images 25 --min-board-span-px 140
```

Collect five views around the center, five near the left, five near the right, five near the top, and five near the bottom, including corner positions. Within each group, change the tilt up/down and left/right by roughly 20–40 degrees and vary the size modestly. Keep all corners visible. Pause before pressing **Space** or **S**; press **Q** when finished. The preview reports corner span and rejects views below the chosen size guide. Position and tilt diversity still require your attention.

After collecting the next set, fit it with the same command using `front_camera_002`, a new candidate path, and a new report path. The tool refuses existing output files unless `--force` is explicitly supplied. If a forced refit is rejected, it saves the rejection report first and removes the old candidate at the explicitly requested output path so it cannot be mistaken for the rejected fit. It does not delete directories or symlink targets.
