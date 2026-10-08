# Camera labels on learned LiDAR boxes

This mode displays native LiDAR car/person points as context and draws boxes only for PointPillars predictions that match a fresh YOLO camera detection, while keeping the black, flat BEV layout. Cars are orange and people are magenta. The LiDAR model supplies center, length/width/height, yaw and its own confidence. Camera labels are associated with the projected LiDAR box and shown separately as `C:<class>` with camera confidence. Box filtering does not change accepted geometry or combine the two model scores.

## Run

After building the project and preparing both saved engines, close an existing RViz window and run:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh --camera-lidar-boxes
```

The wrapper starts or reuses the Ouster driver, native LiDAR vehicle detector, front camera at 1024 × 768 and 30 FPS, native camera detector, and `camera_lidar_box_fusion`. It opens [camera_lidar_boxes.rviz](../config/rviz/camera_lidar_boxes.rviz), showing native orange car points and magenta person points, plus camera-matched flat footprints with predicted dimensions and camera labels. Unmatched boxes are hidden; native points remain visible as context. The labeled front-camera image appears beside the flat BEV, which remains approximately 0–60 m forward at its default size. The camera pane shows YOLO 2D detections; the LiDAR model supplies the 3D dimensions. The separate occupancy window stays closed. RGB cloud projection, ground processing and the older camera-to-point association node are not needed for this mode.

Use `--no-rviz` to start missing processors while keeping a viewer already open. Ctrl+C stops only processes started by the wrapper. Existing camera-only and LiDAR-only preview flags remain available; select one preview mode per command.

Startup checks the required four project executables, both saved engine hashes/class mappings and the DDS receive-buffer prerequisite. It loads [candidate camera calibration B](imported_calibration.md) and the separate [fusion profile](../config/perception/camera_lidar_boxes.yaml). No model compilation occurs during live processing.

## Association and freshness

The fusion node projects the learned LiDAR box using the native camera intrinsics/distortion model and `T_camera_from_os_sensor`, then compares its image bounds with compatible camera detections. Person boxes associate only with camera `person` labels; car boxes use camera vehicle labels. The default association requires camera confidence ≥ 0.35, image-box IoU ≥ 0.20 and an ambiguity margin of 0.05. A nearby or competing image detection can therefore leave a LiDAR box without a camera tag.

Camera detections and CameraInfo must have exactly matching source headers. Camera/LiDAR header differences may be at most 0.15 seconds, and inputs expire after 0.5 seconds. With the default `require_camera_match: true`, missing, empty or stale camera input clears the displayed boxes while native LiDAR car/person points continue independently. Boxes outside the front camera's field of view are hidden, but their native points can remain visible. Missing or stale LiDAR input clears its geometry and points.

The detectors may disagree because they use different sensors and models. A camera person box can appear without a BEV box when PointPillars has no compatible person prediction or projection matching fails; the camera creates no 3D body by itself. The raw `/camera_lidar_bev/lidar_detections_3d` and `/camera_lidar_bev/lidar_vehicle_points` outputs remain unchanged by box filtering. The optional disabled **Camera-matched vehicle/person points** display uses `/camera_lidar_bev/camera_lidar_points` and hides unmatched points too; this topic can be empty while the native points remain populated. To retain unmatched LiDAR boxes as well, set `require_camera_match: false` in the [fusion profile](../config/perception/camera_lidar_boxes.yaml) and restart the fusion node. Unmatched predictions then remain LiDAR-only and carry no camera confirmation or combined score.

An associated camera class is additional model evidence, not independently verified confirmation of the LiDAR detection. Both the camera–LiDAR calibration and timing remain provisional. Overlap alone can associate the wrong image object, especially with occlusion or motion. LiDAR dimensions and yaw remain predictions from the KITTI-trained `Car` and `Pedestrian` classes; a camera class such as `truck` does not make a car prediction trained truck geometry. `Cyclist` remains disabled. See [the LiDAR guide](lidar_vehicle_detection.md) for class coverage, Ouster intensity adaptation and predicted-size limits. Keep the rig stationary for this preview.

## Outputs and performance

| Topic | Message | Meaning |
| --- | --- | --- |
| `/camera_lidar_bev/lidar_detections_3d` | `vision_msgs/Detection3DArray` | Unchanged PointPillars car/person geometry, yaw and LiDAR score |
| `/camera_lidar_bev/camera_lidar_detections_3d` | `vision_msgs/Detection3DArray` | Camera-matched subset retaining exact LiDAR geometry and score |
| `/camera_lidar_bev/detections_2d` | `vision_msgs/Detection2DArray` | Camera class/confidence and image bounds |
| `/camera_lidar_bev/camera_lidar_markers` | `visualization_msgs/MarkerArray` | Matched flat footprints, dimensions and separate camera association labels |
| `/camera_lidar_bev/lidar_vehicle_points` | `sensor_msgs/PointCloud2` | Native orange car and magenta person returns; default point display |
| `/camera_lidar_bev/camera_lidar_points` | `sensor_msgs/PointCloud2` | Returns inside matched boxes; optional filtered point display |
| `/camera_lidar_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Detector and fusion freshness, counters and processing times |

The configured publication cap is 20 Hz. Rates of fresh LiDAR predictions, camera labels and combined marker updates must be measured independently; adding a camera tag does not create a new LiDAR scan. The individual detector checks documented in [camera_objects.md](camera_objects.md) and [lidar_vehicle_detection.md](lidar_vehicle_detection.md) do not establish the combined mode's sustained performance or physical association accuracy.

The updated association replays passed **23/23 phases** in both [retain-unmatched mode](../evaluation/camera_lidar_boxes_replay_002.json) and [require-camera-match mode](../evaluation/camera_lidar_boxes_replay_003.json). They check person/car class compatibility, exact geometry and score preservation, native point provenance, camera failures and stale clearing. The [native detector replay](../evaluation/lidar_vehicle_detector_replay_004.json) passed **14/14 phases**, including trained pedestrian inference on a KITTI reference scan and car inference on the recorded Ouster scan. These are software checks; they do not establish physical pedestrian detection or association accuracy on this rig.

The [updated 10-second live check](../evaluation/camera_lidar_boxes_live_002.json) observed approximately 20 Hz for both detectors and the strict fusion outputs. Native LiDAR produced 3–6 car boxes and 4,760–6,998 selected points per scan, while the camera produced car/truck labels. All strict matched outputs were empty in this scene: publishing updates does not imply a successful association. Native points remained visible in the corrected RViz profile. No live person detection was observed. Ten diagnostic samples measured median full processing of 10.1 ms for LiDAR, 11.0 ms for the camera and 0.34 ms for association. This short observation establishes stream activity, not physical association accuracy or sustained performance.

The historical car-only association core passed eight CPU test groups, and the [earlier isolated ROS replay](../evaluation/camera_lidar_boxes_replay_001.json) passed 16 phases. The [historical 10-second live check](../evaluation/camera_lidar_boxes_live_001.json), with the camera pane running, measured approximately 20 Hz for camera detections, labeled images, LiDAR detections and combined markers. Camera associations appeared on 23 of 200 marker updates in that scene. Ten diagnostic samples measured 0.34 ms median association processing, 9.35 ms median LiDAR node processing and 11.89 ms median camera processing before publication; median combined scan-header age was 163 ms. This live report predates person outputs and strict box filtering. The updated profile still needs its own sustained performance check. Physical association and dimension accuracy remain unvalidated.

For an independently started fusion node after sourcing the ROS overlay, use `ros2 launch bev_bringup camera_lidar_boxes.launch.py`. This starts only the association node; the input sensors and detectors must already be running. Launch arguments `intrinsics_file`, `extrinsics_file` and optional `rviz:=true` select the calibration files and flat viewer.
