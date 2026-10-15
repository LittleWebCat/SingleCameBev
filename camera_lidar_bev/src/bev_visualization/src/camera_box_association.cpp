#include "bev_visualization/camera_box_association.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bev_visualization {
namespace {
bool finite(const XYZ &p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool project_point(const XYZ &sensor_point, const Camera &camera,
                   const Transform &camera_from_sensor, double min_depth,
                   double radial_limit, double &u, double &v) {
  const auto p = transform_point(camera_from_sensor, sensor_point);
  if (!finite(p) || p.z <= min_depth) return false;
  const double x = p.x/p.z, y = p.y/p.z, r2 = x*x+y*y;
  if (!std::isfinite(r2) || r2 >= radial_limit) return false;
  const auto &d = camera.d;
  const double radial = 1+d[0]*r2+d[1]*r2*r2+d[4]*r2*r2*r2;
  const double xd = x*radial+2*d[2]*x*y+d[3]*(r2+2*x*x);
  const double yd = y*radial+d[2]*(r2+2*y*y)+2*d[3]*x*y;
  u = camera.k[0]*xd+camera.k[2];
  v = camera.k[4]*yd+camera.k[5];
  return std::isfinite(u) && std::isfinite(v);
}

ProjectedBox project_validated(const OrientedBox &box, const Camera &camera,
                               const Transform &camera_from_sensor,
                               double min_depth, double radial_limit) {
  ProjectedBox result;
  if (!finite(box.center) || !finite(box.size) ||
      box.size.x <= 0 || box.size.y <= 0 || box.size.z <= 0) return result;
  double norm_squared = 0;
  for (const auto component : box.orientation_xyzw) {
    if (!std::isfinite(component)) return result;
    norm_squared += component*component;
  }
  // Bad quaternions cannot silently produce a different cuboid. Round-off in
  // a valid unit quaternion is normalized before building its rotation matrix.
  if (!std::isfinite(norm_squared) || std::abs(norm_squared-1) > 1e-6) return result;
  const double inverse_norm = 1/std::sqrt(norm_squared);
  const double x = box.orientation_xyzw[0]*inverse_norm;
  const double y = box.orientation_xyzw[1]*inverse_norm;
  const double z = box.orientation_xyzw[2]*inverse_norm;
  const double w = box.orientation_xyzw[3]*inverse_norm;
  const std::array<double, 9> r = {
      1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
  if (!project_point(box.center, camera, camera_from_sensor, min_depth,
                     radial_limit, result.center_u, result.center_v) ||
      result.center_u < 0 || result.center_v < 0 ||
      result.center_u >= camera.width || result.center_v >= camera.height) return {};
  result.x_min = result.y_min = std::numeric_limits<double>::infinity();
  result.x_max = result.y_max = -std::numeric_limits<double>::infinity();
  for (int corner = 0; corner < 8; ++corner) {
    const XYZ local{(corner&1 ? 0.5 : -0.5)*box.size.x,
                    (corner&2 ? 0.5 : -0.5)*box.size.y,
                    (corner&4 ? 0.5 : -0.5)*box.size.z};
    const XYZ sensor{box.center.x+r[0]*local.x+r[1]*local.y+r[2]*local.z,
                     box.center.y+r[3]*local.x+r[4]*local.y+r[5]*local.z,
                     box.center.z+r[6]*local.x+r[7]*local.y+r[8]*local.z};
    double u, v;
    // Conservatively skip boxes crossing the optical near plane or radial
    // fold instead of extrapolating a huge or ambiguous image rectangle.
    if (!project_point(sensor, camera, camera_from_sensor, min_depth,
                       radial_limit, u, v)) return {};
    result.x_min = std::min(result.x_min, u);
    result.y_min = std::min(result.y_min, v);
    result.x_max = std::max(result.x_max, u);
    result.y_max = std::max(result.y_max, v);
  }
  result.x_min = std::max(0.0, result.x_min);
  result.y_min = std::max(0.0, result.y_min);
  result.x_max = std::min(static_cast<double>(camera.width), result.x_max);
  result.y_max = std::min(static_cast<double>(camera.height), result.y_max);
  result.visible = result.x_max > result.x_min && result.y_max > result.y_min;
  return result;
}

bool valid_camera_detection(const CameraDetection &box, const Camera &camera,
                            double min_confidence) {
  if (box.class_name != "person" && box.class_name != "car" && box.class_name != "motorcycle" &&
      box.class_name != "bus" && box.class_name != "truck") return false;
  return std::isfinite(box.confidence) && box.confidence >= min_confidence && box.confidence <= 1 &&
      std::isfinite(box.x_min) && std::isfinite(box.y_min) &&
      std::isfinite(box.x_max) && std::isfinite(box.y_max) &&
      box.x_min >= 0 && box.y_min >= 0 && box.x_max <= camera.width && box.y_max <= camera.height &&
      box.x_max > box.x_min && box.y_max > box.y_min;
}

bool compatible_class(const OrientedBox &lidar, const CameraDetection &camera) {
  if (lidar.class_name == "person") return camera.class_name == "person";
  if (lidar.class_name != "car") return false;
  return camera.class_name == "car" || camera.class_name == "motorcycle" ||
         camera.class_name == "bus" || camera.class_name == "truck";
}

double iou(const ProjectedBox &a, const CameraDetection &b) {
  const double width = std::max(0.0, std::min(a.x_max,b.x_max)-std::max(a.x_min,b.x_min));
  const double height = std::max(0.0, std::min(a.y_max,b.y_max)-std::max(a.y_min,b.y_min));
  const double intersection = width*height;
  const double area_a = (a.x_max-a.x_min)*(a.y_max-a.y_min);
  const double area_b = (b.x_max-b.x_min)*(b.y_max-b.y_min);
  return intersection/(area_a+area_b-intersection);
}

struct Best {
  int index{-1};
  double first{-1}, second{-1};
  void consider(int candidate, double score) {
    if (score > first) {
      second = first; first = score; index = candidate;
    } else if (score > second) {
      second = score;
    }
  }
  bool unambiguous(double margin) const {
    return index >= 0 && (second < 0 || first-second > margin);
  }
};
}  // namespace

void validate_association_config(const AssociationConfig &config) {
  if (!std::isfinite(config.min_iou) || config.min_iou <= 0 || config.min_iou > 1 ||
      !std::isfinite(config.min_confidence) || config.min_confidence < 0 || config.min_confidence > 1 ||
      !std::isfinite(config.ambiguity_margin) || config.ambiguity_margin < 0 || config.ambiguity_margin > 1 ||
      !std::isfinite(config.min_depth_m) || config.min_depth_m <= 0)
    throw std::invalid_argument("Invalid camera/LiDAR box association thresholds");
}

ProjectedBox project_oriented_box(const OrientedBox &box, const Camera &camera,
                                  const Transform &camera_from_sensor, double min_depth_m) {
  validate_camera(camera);
  validate_transform(camera_from_sensor);
  if (!std::isfinite(min_depth_m) || min_depth_m <= 0)
    throw std::invalid_argument("Optical near-plane depth must be finite and positive");
  return project_validated(box, camera, camera_from_sensor, min_depth_m,
                           radial_domain_radius_squared(camera.d));
}

std::vector<BoxCameraMatch> associate_camera_boxes(
    const std::vector<OrientedBox> &lidar_boxes,
    const std::vector<CameraDetection> &camera_boxes,
    const Camera &camera, const Transform &camera_from_sensor,
    const AssociationConfig &config) {
  validate_association_config(config);
  validate_camera(camera);
  validate_transform(camera_from_sensor);
  const auto max_index = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (lidar_boxes.size() > max_index || camera_boxes.size() > max_index)
    throw std::invalid_argument("Box vectors exceed supported association index range");
  const double radial_limit = radial_domain_radius_squared(camera.d);
  std::vector<BoxCameraMatch> result(lidar_boxes.size());
  std::vector<Best> rows(lidar_boxes.size()), columns(camera_boxes.size());
  std::vector<bool> eligible(camera_boxes.size());
  for (std::size_t j = 0; j < camera_boxes.size(); ++j)
    eligible[j] = valid_camera_detection(camera_boxes[j], camera, config.min_confidence);
  for (std::size_t i = 0; i < lidar_boxes.size(); ++i) {
    auto &projection = result[i].projection;
    projection = project_validated(lidar_boxes[i], camera, camera_from_sensor,
                                   config.min_depth_m, radial_limit);
    if (!projection.visible) continue;
    for (std::size_t j = 0; j < camera_boxes.size(); ++j) {
      const auto &box = camera_boxes[j];
      if (!eligible[j] || !compatible_class(lidar_boxes[i],box) ||
          projection.center_u < box.x_min || projection.center_u > box.x_max ||
          projection.center_v < box.y_min || projection.center_v > box.y_max) continue;
      const double overlap = iou(projection, box);
      if (overlap < config.min_iou) continue;
      rows[i].consider(static_cast<int>(j), overlap);
      columns[j].consider(static_cast<int>(i), overlap);
    }
  }
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto &row = rows[i];
    if (!row.unambiguous(config.ambiguity_margin)) continue;
    const auto &column = columns[static_cast<std::size_t>(row.index)];
    if (!column.unambiguous(config.ambiguity_margin) || column.index != static_cast<int>(i)) continue;
    result[i].camera_index = row.index;
    result[i].iou = row.first;
  }
  return result;
}

}  // namespace bev_visualization
