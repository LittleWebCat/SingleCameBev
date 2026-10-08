#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bev_lidar_detection {

// KITTI LiDAR coordinates: x forward, y left, z up, meters. Intensity [0,1].
struct PointXYZI { float x, y, z, intensity; };
struct Box3D {
  float x, y, z;
  float size_x, size_y, size_z;
  float yaw, score;
  int class_id;  // 0 Car, 1 Pedestrian, 2 Cyclist; not COCO IDs.
};
struct DetectorStats {
  double processing_ms = 0;
  std::size_t input_points = 0;
  std::size_t accepted_points = 0;
  std::size_t input_pillars = 0;
  std::size_t accepted_pillars = 0;
  std::size_t truncated_pillars = 0;
  std::size_t output_boxes = 0;
};

class Detector {
 public:
  static constexpr std::size_t max_points = 300000;
  static constexpr std::size_t max_pillars = 10000;
  Detector(const std::string &engine_path, float score_threshold = 0.35f,
           std::size_t max_output = 200);
  ~Detector();
  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;
  std::vector<Box3D> infer(const std::vector<PointXYZI> &points);
  const DetectorStats &lastStats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace bev_lidar_detection
