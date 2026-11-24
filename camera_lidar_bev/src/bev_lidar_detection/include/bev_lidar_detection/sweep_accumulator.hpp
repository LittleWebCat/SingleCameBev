#pragma once

#include "bev_lidar_detection/detector.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace bev_lidar_detection {

// Pose of os_sensor in a fixed world/map frame: translation (m) and unit quaternion (x, y, z, w).
struct SensorPose {
  std::array<double, 3> t{0, 0, 0};
  std::array<double, 4> q{0, 0, 0, 1};
};

// Keeps the most recent sweeps with their poses and merges them, nuScenes style, into the
// newest sweep's os_sensor frame. Each merged point carries its sweep's age in seconds
// (newest sweep 0), which is CenterPoint's time-lag input feature.
class SweepAccumulator {
 public:
  // max_sweeps includes the newest one; sweeps older than max_span_s are dropped.
  SweepAccumulator(std::size_t max_sweeps, double max_span_s);

  // Adds a sweep (points in its own os_sensor frame) stamped at stamp_ns, the time its pose
  // refers to. A stamp that is not newer than the last sweep resets the history.
  void add(std::vector<PointXYZI> points, std::int64_t stamp_ns, const SensorPose &world_from_sensor);

  // Merged cloud in the newest sweep's frame and per-point ages. Empty if no sweeps.
  void build(std::vector<PointXYZI> &points, std::vector<float> &ages) const;

  // Replaces the stored pose of the sweep stamped within 1 ms of stamp_ns (e.g. a predicted
  // pose by the odometry's own estimate once it arrives). Returns false if none matches.
  bool refine_pose(std::int64_t stamp_ns, const SensorPose &world_from_sensor);

  void clear() { sweeps_.clear(); }
  std::size_t sweeps() const noexcept { return sweeps_.size(); }

 private:
  struct Sweep {
    std::vector<PointXYZI> points;
    std::int64_t stamp_ns;
    SensorPose pose;
  };
  std::size_t max_sweeps_;
  std::int64_t max_span_ns_;
  std::deque<Sweep> sweeps_;
};

}  // namespace bev_lidar_detection
