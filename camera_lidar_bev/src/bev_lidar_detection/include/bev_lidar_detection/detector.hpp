#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bev_lidar_detection {

// os_sensor coordinates: x forward, y left, z up, meters. Intensity [0,1].
struct PointXYZI { float x, y, z, intensity; };
struct Box3D {
  float x, y, z;
  float size_x, size_y, size_z;
  float yaw, score;
  int class_id;  // Index into ObjectDetector::classNames(); not COCO IDs.
};
struct DetectorStats {
  double processing_ms = 0;
  double preprocess_ms = 0;
  double inference_ms = 0;
  double postprocess_ms = 0;
  std::size_t input_points = 0;
  std::size_t accepted_points = 0;
  std::size_t input_pillars = 0;
  std::size_t accepted_pillars = 0;
  std::size_t truncated_pillars = 0;
  std::size_t truncated_points = 0;
  std::size_t output_boxes = 0;
};

class ObjectDetector {
 public:
  virtual ~ObjectDetector() = default;
  virtual std::vector<Box3D> infer(const std::vector<PointXYZI> &points) = 0;
  virtual const DetectorStats &lastStats() const noexcept = 0;
  // Output class names, already mapped to this project's vocabulary ("car", "person", ...).
  virtual const std::vector<std::string> &classNames() const noexcept = 0;
  virtual std::size_t maxPoints() const noexcept = 0;
  virtual const char *modelName() const noexcept = 0;
};

// NVIDIA CUDA-PointPillars, KITTI weights: forward-only 0..69.12 m.
class Detector final : public ObjectDetector {
 public:
  static constexpr std::size_t max_points = 300000;
  static constexpr std::size_t max_pillars = 10000;
  Detector(const std::string &engine_path, float score_threshold = 0.35f,
           std::size_t max_output = 200);
  ~Detector() override;
  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;
  std::vector<Box3D> infer(const std::vector<PointXYZI> &points) override;
  const DetectorStats &lastStats() const noexcept override;
  const std::vector<std::string> &classNames() const noexcept override;
  std::size_t maxPoints() const noexcept override { return max_points; }
  const char *modelName() const noexcept override { return "PointPillars_KITTI_FP16"; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace bev_lidar_detection
