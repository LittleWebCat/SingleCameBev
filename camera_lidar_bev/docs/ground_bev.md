# Stationary ground, obstacles and occupancy

The C++ `bev_geometry` package consumes the existing `/camera_lidar_bev/points_rgb` in `os_sensor`. It estimates a supported floor, classifies measured returns and publishes a conservative grid. The height prior is the user's approximate **1.8 m** sensor mounting height; the fitted floor is estimated from each scan.

## Run

After building, one command starts missing project processes and opens the ground and occupancy views:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh
```

The wrapper sources the ROS environments, checks the receive-buffer prerequisite, and reuses existing camera, Ouster, RGB and ground processes in the selected ROS domain. It refuses duplicate processors and conflicting domains. A missing receive-buffer setting reports `sudo bash tools/setup_dds_transport.sh` for persistent setup, plus a temporary sysctl command; the launcher does not run sudo itself. Ctrl+C stops only the processes started by this invocation. Existing sensor processes stay owned by their original launchers.

Use `--dry-run` to inspect startup without launching anything, `--no-rviz` to keep an existing RViz view, and `--no-occupancy-view` to skip the grid window. To open only the grid while its processor is running:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/view_occupancy.py
```

For direct processor bring-up, `ros2 launch bev_bringup ground_bev.launch.py` consumes the already-running RGB cloud. Set `rviz:=false` to run only that processor.

## Interpret the views

| Color | Point-cloud view | Occupancy image |
| --- | --- | --- |
| Green | Measured return near supported local ground | Cell with observed ground and unobstructed 2D visibility |
| Red | Return 0.15–3.0 m above supported local ground | Cell containing obstacle evidence |
| Gray | No reliable local ground classification | Unobserved, sparse, blocked or unsupported cell |

The grid covers **40 m forward × 30 m wide** at **0.20 m per cell**. The native image is 150 pixels wide and 200 pixels high; forward points upward and left points leftward. Every image pixel represents one grid cell. The viewer scales cells with nearest-neighbor interpolation and adds the legend without filling empty data.

The ROS `OccupancyGrid` uses x-fastest row order, whereas `bev_core` uses lateral-fastest storage. The adapter converts explicitly; tests verify the image axes and every color against the ROS grid.

These outputs are in the **LiDAR sensor frame**. The sensor-to-vehicle transform remains unmeasured, so they are not labeled `base_link`.

## Ground and visibility evidence

Low spatial patches propose a floor consistent with the configured height and slope limits. Height-guided two-point proposals improve fit stability when floor returns are outnumbered by walls or roofs. A candidate needs a connected spatial component with enough patches, returns and two-dimensional coverage. Refinement uses equally weighted patch seeds; its RMS describes those fitting seeds rather than dense wall points near the tolerance boundary.

Local corrections are bounded to half the ground tolerance. No supported floor produces unknown labels and an unknown grid. The core does not reuse an earlier plane when the current scan is unsupported.

Obstacle cells are established first. A free cell needs a directly observed ground endpoint and a clear 2D line of sight; occupied cells stop visibility, including corner-touching rays. Sparse gaps and regions behind obstacles remain unknown. This deliberately conservative grid can stay mostly unknown even when elevated beams see ground beyond a barrier.

Vehicle/platform exclusion is **enabled** using the user's body dimensions and LiDAR position. The mask covers `x = [-2.131, 2.350] m`, `y = [-0.905, 0.885] m`, and `z = [-1.800, 0.000] m` in `os_sensor`: 2.35 m ahead, 2.131 m behind, 0.885 m left and 0.905 m right, below the sensor. The rear extent follows from the reported 4.481 m body length. These measurements and their provenance are saved in [vehicle_self_mask_001.json](../evaluation/vehicle_self_mask_001.json).

The mask excludes measured returns inside the body/mount envelope from floor fitting, clouds and obstacle evidence. It does not declare the footprint free. Mirrors remain outside the mask because their positions relative to the LiDAR have not been measured; the approximate mirror-to-mirror width does not define a safe exclusion box. This sensor-frame box does not establish a calibrated sensor-to-vehicle transform.

All outputs describe one scan. They do not accumulate a rolling map or deskew movement. Keep the rig stationary. Camera calibration, motion and exposure synchronization remain provisional. Old inputs clear the clouds, grid and image; the grid viewer also clears its display if image updates stop.

## Configuration and validation

[ground_bev.yaml](../config/perception/ground_bev.yaml) sets the height prior, slope/support limits, local ground radius, obstacle height interval and grid bounds. The node publishes separate ground, obstacle and unknown clouds, a combined diagnostic cloud, `/camera_lidar_bev/occupancy`, and `/camera_lidar_bev/occupancy_image`. Cloud encoding is skipped for outputs without subscribers.

Core regressions cover slopes, boxes, curbs, walls, disconnected roofs, unsupported scans, range limits, measured masks and obstacle shadows. The bounded [ROS verifier](../tools/check_ground_bev.py) uses private topics in an isolated domain to check point conservation, original source indices, colors, height evidence, grid states, exact image orientation, wrong frames and stale clearing.

The first [recording report](../evaluation/ground_baseline_recording_001.json) contains a 4.607-second camera/LiDAR recording. It has 51 saved raw scans and 92 image/CameraInfo pairs; the additional recorder subscription reduced delivered rates, so this bag is not a sustained 20 Hz transport gate. A separate fresh colorized-cloud fixture is in `data/geometry/ground_baseline_001`.

The [current isolated replay](../evaluation/ground_bev_replay_003.json) passed 52 classified outputs across flat ground, slopes, unsupported floor, the saved fixture, recovery and the measured vehicle mask. It verifies every image pixel against the grid, clears stale/wrong-frame evidence, and keeps nearby obstacle points just outside each mask boundary. All six core regression groups pass.

The [current live check](../evaluation/ground_bev_live_002.json) observed 237 matching grid/image frames over 12 seconds, approximately **19.7 Hz**, while RViz and the occupancy window were open. Every matched frame contained supported evidence; all 12 sampled ground diagnostics reported a valid floor. The configured body mask removed 1,559–1,596 points per sampled scan, and three inspected clouds contained no points inside that box. Median sampled processing time was 29.6 ms, with a 96.8 ms maximum. This short observation is not a sustained performance gate. The [RViz snapshot](../evaluation/ground_bev_live_002.png) and [occupancy snapshot](../evaluation/occupancy_live_001.png) show the current views.

The last observed grid contained 3 free, 504 occupied and 29,493 unknown cells. The small free region reflects the conservative visibility rule and the current occluding scene; gray cells are not free space. Synthetic checks and filter-bound verification do not replace a known-distance physical obstacle test, which remains pending. The [earlier live report](../evaluation/ground_bev_live_001.json) records the initial implementation's roughly 10 Hz refresh before fit and serialization improvements.

To repeat the bounded software check without changing live hardware:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
/usr/bin/python3 tools/check_ground_bev.py --check-vehicle-mask \
  --fixture data/geometry/ground_baseline_001 \
  --output evaluation/ground_bev_replay_new.json
```

Choose a new evidence filename on each run; the checker refuses to replace existing reports.
