# Front-camera intrinsic calibration review: second dataset

**Result: candidate intrinsics applied provisionally for live bring-up.** The 25 closer-board images substantially improve coverage and focal-length stability. After viewing the live corrected image, the user reported that a straight edge stays straight. The [candidate YAML](../config/calibration/front_camera_640x480_002.candidate.yaml) is now loaded by the live camera bridge at 640 × 480. Independent numerical accuracy checks remain pending.

The board is the user-confirmed printed pattern with 6 × 5 total squares, 5 × 4 inner corners, and 65 mm square edges. Images were captured at 640 × 480. Physical print spacing and board flatness remain assumptions; the photos cannot independently establish them.

## Evidence

The offline helper re-detected all 20 corners in all 25 PNGs. The [detailed JSON report](front_camera_002_fit.json) preserves source hashes, the exact split, fitted coefficients, uncertainty, and stability checks. An independent audit reproduced the fit and verified all 25 unique source hashes against the raw files.

| Check | Second dataset |
| --- | --- |
| Training / held-out images | 20 / 5, selected before fitting |
| Held-out frames | 0003, 0008, 0013, 0018, 0023 |
| Training reprojection RMS | 0.629 pixels |
| Held-out pooled reprojection RMS | 0.611 pixels |
| Maximum held-out view RMS | 0.701 pixels |
| Median inner-corner span, all images | 319 × 242 pixels |
| Combined corner convex-hull area, all images | 56.7% of image area |
| Spatial bins containing training corners | 12 of 12 in a 4 × 3 grid |
| Maximum focal-length change in three training subset fits | 2.09% |
| Maximum principal-point change in those subset fits | 16.1 pixels |
| Focal-length change after removing the largest training board view | 0.415% |
| Minimum radial scale / mapping derivative in nominal image domain | 0.922 / 0.736, both positive |

![Comparison of observed corner coverage](front_camera_002_coverage.png)

The folding failure from [the first dataset](front_camera_001_review.md) is resolved for this candidate's nominal image domain. Focal lengths are much less sensitive to view selection. Principal-point sensitivity still triggers a heuristic warning, and the held-out error measures reprojection consistency with independently fitted board poses, not ground-truth intrinsic accuracy.

The candidate matrix has fx = 904.89, fy = 894.19, cx = 348.90, and cy = 239.30 pixels. Its `plumb_bob` coefficients in OpenCV/ROS order are approximately `[-0.407965, 0.808052, 0.004596, 0.000965, -2.542243]`. Use the full-precision YAML rather than copying rounded values. R is identity and P is `[K|0]` for the monocular camera; neither is a measured camera–LiDAR transform.

## Controlled undistortion preview

The preview compares the same received image on the left and right. It uses the candidate at native dimensions with the same K, without cropping or implicitly scaling the calibration. The label keeps its unvalidated status visible. A held-out image comparison is saved below:

![Held-out raw image and candidate undistortion](front_camera_002_undistortion.jpg)

To open the live comparison with the camera bridge already running:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=1 DISPLAY=:1 /usr/bin/python3 tools/view_camera.py \
  --calibration-file config/calibration/front_camera_640x480_002.candidate.yaml
```

Press **Q** or **Esc** to close. To save one comparison without a GUI, add `--snapshot /tmp/front_camera_candidate_comparison.png`. Previewing does not alter ROS camera information or the stored calibration images.

## Visual check and provisional application

The user reported **“it stays straight”** after the live raw/corrected comparison. This is a qualitative check; no numerical straightness residual or per-position measurement was recorded. The bridge was restarted with the explicit candidate path and the live stream verified. The [application evidence](front_camera_002_application.json) records 59 matched image/CameraInfo headers, the full K/D/R/P match, one publisher per topic, and two timing warnings with no camera error.

`CameraInfo.k[0]` is now 904.886360419207 rather than zero. The camera diagnostic's `calibrated: true` means that intrinsics are loaded and match the actual image dimensions; it does not qualify numerical accuracy or multi-sensor calibration. The diagnostic retains WARN for unavailable exposure timestamps and unverified synchronization. The fitting report and YAML comments preserve the original state at fitting time; the application evidence records the later live state.

For a future restart, use the explicit path after sourcing ROS and this overlay:

```bash
ROS_DOMAIN_ID=1 ros2 launch bev_bringup front_camera.launch.py \
  calibration_file:=/home/agxthor/SingleCameraBev/camera_lidar_bev/config/calibration/front_camera_640x480_002.candidate.yaml
```

The default launch still selects an empty calibration file. To repeat the bounded live check:

```bash
ROS_DOMAIN_ID=1 /usr/bin/python3 tools/check_camera_calibration.py
```

It checks paired headers, dimensions, encoding, camera matrices, publisher counts, and timing diagnostics. The default evidence output is `/tmp/front_camera_calibration_check.json`.

## Remaining accuracy and alignment checks

Show a straight ruler, door frame, or another independently straight edge near the left, right, top, and bottom of the view, including corners. Inspect whether the corrected pane introduces bending or obvious stretching. Check the printed target is flat and rigid and that spacing is 65 mm in both directions. At this square size, the outer printed grid should measure 390 × 325 mm; the inner-corner extents are 260 × 195 mm. Keep the operating resolution and lens settings consistent.

A favorable visual check supports further evaluation; it does not alone establish a numerical accuracy bound. Keep the candidate status until independent accuracy requirements are met. Camera–LiDAR alignment is the next geometry step, using sensors held rigidly in their operating positions and a target observed by both. Vehicle transforms and synchronization also remain unmeasured.

## Software verification

The candidate YAML's K/D match the report and all matrices are finite and structurally compatible with ROS. The actual installed C++ camera bridge accepted the YAML and published two synthetic frames in isolated ROS domain 189 without opening a camera device. This establishes file compatibility, not sensor accuracy. The new preview helper's seven tests passed, covering distorted ray geometry, raw pixel preservation, malformed YAML, and resolution mismatch. The actual dataset rendered successfully in the preview helper. The live comparison window was verified on the desktop; a received 640 × 480 image also produced a correctly sized 1280 × 516 comparison snapshot.
