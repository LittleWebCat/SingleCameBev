# Archived camera-only 3D detection experiment

**Status: removed from the active pipeline after the camera-only 3D results were rejected.** Normal live, replay and combined views no longer launch or display SMOKE predictions. The camera supplies **YOLO 2D object labels and TwinLiteNet road/lane segmentation**; **LiDAR supplies the 3D boxes**. Camera labels receive a BEV position through an accepted camera–LiDAR association.

The SMOKE source, saved model, provenance and historical results remain available for reference. This document describes that archived experiment; it provides no active camera-only 3D startup instructions. Use [the combined display guide](combined_bev_display.md) for the current pipeline.

The experiment used [SMOKE monocular 3D detection](https://github.com/lzccccc/SMOKE) through native C++/TensorRT to estimate car/person depth, center, dimensions and heading from a camera image and intrinsics, independently of LiDAR or a YOLO-to-LiDAR match.

## Historical saved outdoor example

The saved TensorRT result on the calibrated KITTI outdoor frame `000008` contains **seven car predictions**; see [the historical result report](../evaluation/camera_3d_saved_example_001.json). Its annotated image, `evaluation/camera_3d_saved_example_001.png`, is a local artifact excluded from source control. This example used the frame's KITTI calibration and no LiDAR. It demonstrated execution of the pretrained model, without establishing detection accuracy or ground-truth size accuracy on this rig.

The downloaded input image/calibration, saved checkpoint and engine remain local. The source repository includes provenance and result JSON. The offline preview helper `tools/preview_camera_3d_frame.py` is retained as experimental reference code.

## Historical inputs and coordinates

The experiment consumed `/camera/front/image_raw` and its paired `/camera/front/camera_info`, using the existing native [candidate intrinsics](../config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml). The archived [profile](../config/perception/camera_3d_detector.yaml) enabled `car` and `person`, confidence ≥ 0.25, a 0.5 s freshness limit and a 20 Hz publication cap. `Cyclist` was available in the trained model but disabled in this profile.

Its 3D detection output used `camera_front_optical_frame`: X right, Y down, Z forward. Standalone flat markers used a camera-relative `camera_front_bev` frame with axes `X = optical Z`, `Y = −optical X`, `Z = −optical Y`. Its origin was the camera. This coordinate conversion established no `base_link` calibration, odometry or camera–LiDAR transform, and did not establish aligned LiDAR-map objects.

## Historical model and outputs

The saved engine is `models/camera_3d_001/smoke_fp16_raw.engine`. [Model provenance](../models/camera_3d_001/provenance.json) records the SMOKE source, saved weights, engine hash, class order and tensor contract. The native input was FLOAT `[1, 3, 384, 1280]`; raw heads were a three-class heatmap `[1, 3, 96, 320]` and eight-channel regression `[1, 8, 96, 320]`. The engine was loaded once at detector startup.

These topics document the historical detector's output contract. The normal live/replay/combined pipeline no longer publishes or subscribes to them:

| Historical topic | Message | Meaning |
| --- | --- | --- |
| `/camera_lidar_bev/camera_detections_3d` | `vision_msgs/Detection3DArray` | Monocular predicted geometry and score in the camera optical frame |
| `/camera_lidar_bev/camera_3d_labeled` | `sensor_msgs/Image` | Camera image annotated with the model's detections |
| `/camera_lidar_bev/camera_3d_markers` | `visualization_msgs/MarkerArray` | Flat camera-relative footprints and predicted dimensions |

## Historical limitations and measurements

Depth, length/width/height and heading were learned **estimates**, including hidden surfaces. The model was trained on KITTI, and this camera differs in lens, calibration, scene and image distribution. Accuracy and metric scale on this rig were not established. Its trained classes are `Car`, `Cyclist` and `Pedestrian`; it has no trained bus/truck class. A camera-only box could be missed, misplaced or incorrectly sized even when its image overlay looked plausible.

The rig's native focal length is approximately 1411 pixels, compared with approximately 721 pixels for the KITTI example. Native intrinsics were used for geometric backprojection, but learned depth was not normalized for this focal-length difference. A plausible overlay did not validate physical range on this camera.

The [model manifest](../models/camera_3d_001/provenance.json) records TensorRT-versus-reference numerical parity on the seven-car example: maximum center difference 0.0555 m, dimension difference 0.00355 m, heading difference 0.00201 radians and projected-corner difference 0.239 pixels. These compare implementations of the same model; they are not errors against ground truth.

Historical engine-only benchmarking measured **25.50 ms median** and **27.65 ms p95** over 30 runs after five warmups, approximately 39 engine inferences per second. That measurement excluded capture, preprocessing, decoding, ROS publication and display. At implementation time, the native C++ build and five CTest targets passed, including six camera-3D geometry groups; these results established software checks, not physical accuracy.

The later [saved-frame perception comparison](../evaluation/perception_speed_compare_001.json) measured camera-only processing at **47.8 ms median / 49.9 ms p95**, versus **15.5 ms median / 16.4 ms p95** for LiDAR + YOLO + association. Those timings excluded capture/loading and display; LiDAR point conversion happened before timing. They are historical measurements, not active-pipeline throughput claims.

The 20 Hz publication cap was a configured target. Physical range and dimension accuracy were not established, and this experiment is no longer part of the project's active detection pipeline.
