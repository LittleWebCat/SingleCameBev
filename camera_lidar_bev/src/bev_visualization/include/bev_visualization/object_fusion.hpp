#pragma once

#include "bev_visualization/rgb_projection.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bev_visualization {

// Bounds are in the calibrated native raw image, before any detector resize.
struct CameraDetection {
  std::string class_name;
  double confidence{};
  double x_min{}, y_min{}, x_max{}, y_max{};
};

struct ClassifiedPoint {
  XYZ point{};
  std::uint32_t source_index{};
  std::uint8_t classification{};  // Ground=2, Obstacle=3, Unknown=1.
  double height_above_ground{};
};

struct ObjectCandidate {
  std::string class_name;
  double confidence{};  // Camera confidence, not calibrated 3D confidence.
  XYZ center{}, size{};  // Axis-aligned extent of observed LiDAR returns only.
  // Rows in the supplied classified vector; original source indices survive there.
  std::vector<std::size_t> classified_indices;
  std::size_t visible_support{};
};

struct FusionConfig {
  double min_confidence{0.35};
  std::size_t min_support_points{8};
  std::size_t min_visible_pixels{6};
  std::size_t min_central_points{2};
  double cluster_resolution_m{0.45};  // Euclidean adjacency radius; spatial hash cell size.
  double min_optical_depth_m{0.3};
  double max_optical_depth_m{60.0};
  double max_observed_extent_m{12.0};
  double max_observed_height_m{4.5};
  double ambiguity_depth_m{0.90};
  double ambiguity_support_ratio{0.65};
  std::size_t max_detections{100};
  std::size_t max_points_per_detection{20000};
};

struct FusionResult {
  std::vector<ObjectCandidate> candidates;
  std::size_t detections_considered{}, visible_geometry_points{};
  std::size_t supported_obstacle_points{};
  std::size_t rejected_invalid_detections{}, rejected_unsupported{}, rejected_ambiguous{};
};

void validate_fusion_config(const FusionConfig &config);

// Full geometry includes the vehicle/mount returns omitted by classification.
// Both geometry arrays use identical row order; IDs identify original laser returns.
// The caller pairs image detections and these clouds by sensor timestamps.
// Nothing is inferred behind an occluder, without supported obstacle evidence, or
// outside camera coverage. Geometry and occupancy are never modified.
FusionResult fuse_camera_objects(
    const std::vector<XYZ> &full_geometry,
    const std::vector<std::uint32_t> &full_source_indices,
    const std::vector<ClassifiedPoint> &classified,
    const std::vector<CameraDetection> &detections,
    const Camera &camera, const Transform &camera_from_sensor,
    const FusionConfig &config = {});

}  // namespace bev_visualization
