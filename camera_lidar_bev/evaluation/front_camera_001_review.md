# Front-camera intrinsic calibration review: first dataset

**Result: rejected lens model; camera remains uncalibrated.** All 25 source images are preserved in `data/calibration/front_camera_001`. No loadable camera calibration YAML was produced or applied to the bridge.

The user confirmed one printed checkerboard, 6 × 5 total squares with 65 mm square edges. The model uses 5 × 4 inner corners. Board flatness, printing accuracy, and physical square spacing were not independently measured.

## Saved evidence and numerical checks

The offline helper re-detected all 20 corners in all 25 raw PNGs at 640 × 480. The [JSON report](front_camera_001_fit.json) records image hashes, exact training/held-out membership, fitted values, uncertainty estimates, and sensitivity diagnostics.

| Check | Observed result |
| --- | --- |
| Training / held-out views | 20 / 5, selected before fitting |
| Held-out images | Frames 0003, 0008, 0013, 0018, 0023 |
| Training reprojection RMS | 0.281 pixels |
| Held-out pooled reprojection RMS | 0.289 pixels |
| Largest held-out view RMS | 0.538 pixels |
| Median inner-corner span across all saved views | 38 × 30 pixels |
| Observed corner vertical range across all saved views | y = 201–350 pixels of a 480-pixel image |
| Combined corner convex-hull area across all saved views | 15.6% of image area |
| Change in fitted fx/fy after removing the largest training view | −11.7% / −11.1% |
| Principal-point change after removing that view | −23.5 / −46.5 pixels |
| Maximum focal change in three training subset refits | 69.9% |
| Minimum radial-mapping derivative in nominal image domain | −5.01; mapping folds |

![Corner coverage and board sizes](front_camera_001_coverage.png)

The low reprojection error does not resolve the coverage and stability problems. Each held-out view has its own fitted board pose, which can absorb intrinsic errors. Most boards are small and observations cover a narrow horizontal band. The trial fit is sensitive to which training views are included.

The radial mapping derivative reaches zero at normalized radius 0.385 and becomes negative before the conservative nominal image-domain limit of 0.493. This indicates an invalid extrapolation for full-frame use. The diagnostic domain comes from the fitted pinhole matrix; it is not a measured field of view. OpenCV's [calibration documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html) explains that non-monotonic distortion indicates calibration failure even when the center appears well fitted.

## Next capture

The next live capture window uses `data/calibration/front_camera_002`, a 25-image target, and `--min-board-span-px 140`. The 140-pixel guide applies to the larger x/y inner-corner span and rejects very small boards. It is a collection heuristic, not a calibration acceptance threshold.

Move closer so the whole board occupies roughly one-third to two-thirds of the image width. Collect views at the center, left, right, top, bottom, and corner positions, with different tilts in both axes. Keep the printed board flat and rigid, all corners visible, and pause before saving. The next dataset must be fitted and reviewed again before loading calibration.

Reproduction and commands are in [camera_calibration_fit.md](../docs/camera_calibration_fit.md). The helper returned exit code 2 and wrote the rejection report without a YAML. Its 15 numerical and output tests passed, including known-intrinsics recovery, held-out corner errors, degenerate frontal views, radial folding, and removal of stale candidate outputs after forced rejected refits. The capture helper's 13 tests passed, including the new pixel-span guide. The live capture window was verified on the desktop.
