#pragma once

#include "bev_visualization/object_fusion.hpp"

#include <array>
#include <string>
#include <vector>

namespace bev_visualization {

// Complete predicted LiDAR geometry in os_sensor. Quaternion order is x,y,z,w.
struct OrientedBox {
  XYZ center{}, size{};
  std::array<double, 4> orientation_xyzw{0, 0, 0, 1};
  std::string class_name{"car"};
};

struct ProjectedBox {
  bool visible{false};
  double x_min{}, y_min{}, x_max{}, y_max{}, center_u{}, center_v{};
};

struct AssociationConfig {
  double min_iou{0.20};
  double min_confidence{0.35};
  double ambiguity_margin{0.05};
  double min_depth_m{0.30};
};

// One result per input LiDAR box. The camera index addresses the original input
// camera vector; -1 means unmatched. Scores and complete LiDAR geometry remain
// separate: projected overlap supplies an association, not occlusion proof.
struct BoxCameraMatch {
  int camera_index{-1};
  double iou{0};
  ProjectedBox projection{};
};

void validate_association_config(const AssociationConfig &config);

// Uses the calibrated native raw image, including plumb_bob distortion. Boxes
// with an invalid quaternion, any corner crossing the optical near plane or
// central distortion domain, or an out-of-view center remain invisible.
// Global calibration/configuration errors throw std::invalid_argument.
ProjectedBox project_oriented_box(const OrientedBox &box, const Camera &camera,
                                  const Transform &camera_from_sensor,
                                  double min_depth_m = 0.30);

// Unique mutual-best matches only. Near ties in either direction are rejected.
// LiDAR car boxes accept vehicle camera classes; person boxes accept person
// only. Other LiDAR classes remain unmatched. Camera classes are separate
// evidence and never alter predicted box geometry.
std::vector<BoxCameraMatch> associate_camera_boxes(
    const std::vector<OrientedBox> &lidar_boxes,
    const std::vector<CameraDetection> &camera_boxes,
    const Camera &camera, const Transform &camera_from_sensor,
    const AssociationConfig &config = {});

}  // namespace bev_visualization
