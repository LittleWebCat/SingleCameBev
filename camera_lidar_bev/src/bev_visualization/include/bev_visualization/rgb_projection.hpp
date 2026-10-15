#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bev_visualization {

using Transform = std::array<double, 16>;  // Row-major, translation in meters.
struct XYZ { double x, y, z; };
struct Camera {
  std::uint32_t width{}, height{};
  std::array<double, 9> k{};
  std::array<double, 5> d{};  // k1, k2, p1, p2, k3 (plumb_bob).
};
struct ImageView {
  const std::uint8_t *data{};
  std::size_t size{}, step{};
  std::uint32_t width{}, height{};
  bool bgr{true};
};
struct ColoredPoint {
  float x{}, y{}, z{};
  std::uint32_t rgb{0x00ffffff};
  std::uint32_t source_index{};
  bool camera_colored{false};
};
struct ColorResult {
  std::vector<ColoredPoint> points;
  std::size_t input_count{}, front_count{}, radial_excluded_count{}, in_frame_count{}, colored_count{};
};

struct VisibleProjection {
  std::size_t point_index{};  // Row in the supplied os_sensor geometry.
  double u{}, v{}, optical_depth{};
  std::uint32_t pixel_x{}, pixel_y{};
};
struct ProjectionResult {
  std::vector<VisibleProjection> points;
  std::size_t input_count{}, front_count{}, radial_excluded_count{}, in_frame_count{};
};

Transform identity_transform();
void validate_transform(const Transform &transform);
Transform compose(const Transform &left, const Transform &right);
XYZ transform_point(const Transform &transform, const XYZ &point);
void validate_camera(const Camera &camera);
double radial_domain_radius_squared(const std::array<double, 5> &distortion);
// Project os_sensor geometry into the calibrated native raw image. One return
// survives per native pixel: nearest optical depth, then earliest input row.
// Points outside the central radial domain remain unassociated.
ProjectionResult project_visible_sensor_points(const std::vector<XYZ> &sensor_points,
                                               const Camera &camera,
                                               const Transform &camera_from_sensor);
// Every finite, nonzero source return survives as geometry in os_sensor.
// Only the nearest optical-depth return at each native distorted image pixel
// receives RGB. Other points are white. Equal-depth ties keep the first source index.
ColorResult colorize(const std::vector<XYZ> &source_points, const Camera &camera,
                     const ImageView &image, const Transform &camera_from_sensor,
                     const Transform &sensor_from_lidar);

}  // namespace bev_visualization
