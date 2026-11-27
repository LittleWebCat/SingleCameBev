#include "bev_visualization/rgb_projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace bev_visualization {
namespace {
bool finite(const XYZ &p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }
double first_positive_root(std::vector<double> coefficients) {
  while (coefficients.size() > 1 && coefficients.back() == 0.0) coefficients.pop_back();
  if (coefficients.size() == 1) return std::numeric_limits<double>::infinity();
  cv::Mat roots;
  cv::solvePoly(cv::Mat(coefficients), roots);
  double result = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < roots.total(); ++i) {
    const auto root = roots.at<cv::Vec2d>(static_cast<int>(i));
    if (root[0] > 0.0 && std::abs(root[1]) <= 1e-7 * std::max(1.0, std::abs(root[0]))) {
      result = std::min(result, root[0]);
    }
  }
  return result;
}
}  // namespace

Transform identity_transform() { return {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; }

void validate_transform(const Transform &t) {
  for (const auto value : t) if (!std::isfinite(value)) throw std::invalid_argument("Transform is nonfinite");
  constexpr double tolerance = 1e-8;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(t[12 + i] - (i == 3 ? 1.0 : 0.0)) > tolerance)
      throw std::invalid_argument("Transform has invalid homogeneous row");
  }
  for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
    double dot = 0.0;
    for (int k = 0; k < 3; ++k) dot += t[k * 4 + row] * t[k * 4 + column];
    if (std::abs(dot - (row == column ? 1.0 : 0.0)) > tolerance)
      throw std::invalid_argument("Transform rotation is not orthogonal");
  }
  const double det = t[0] * (t[5] * t[10] - t[6] * t[9]) -
      t[1] * (t[4] * t[10] - t[6] * t[8]) + t[2] * (t[4] * t[9] - t[5] * t[8]);
  if (std::abs(det - 1.0) > tolerance) throw std::invalid_argument("Transform rotation is not proper");
}

Transform compose(const Transform &left, const Transform &right) {
  validate_transform(left); validate_transform(right);
  Transform result{};
  for (int row = 0; row < 4; ++row) for (int column = 0; column < 4; ++column)
    for (int k = 0; k < 4; ++k) result[row * 4 + column] += left[row * 4 + k] * right[k * 4 + column];
  return result;
}

XYZ transform_point(const Transform &t, const XYZ &p) {
  return {t[0]*p.x + t[1]*p.y + t[2]*p.z + t[3],
          t[4]*p.x + t[5]*p.y + t[6]*p.z + t[7],
          t[8]*p.x + t[9]*p.y + t[10]*p.z + t[11]};
}

void validate_camera(const Camera &c) {
  if (!c.width || !c.height || c.width > 16384 || c.height > 16384)
    throw std::invalid_argument("Invalid native image dimensions");
  for (const auto v : c.k) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite K");
  for (const auto v : c.d) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite D");
  if (c.k[0] <= 0 || c.k[4] <= 0 || std::abs(c.k[1]) > 1e-12 || std::abs(c.k[3]) > 1e-12 ||
      std::abs(c.k[6]) > 1e-12 || std::abs(c.k[7]) > 1e-12 || std::abs(c.k[8] - 1.0) > 1e-12)
    throw std::invalid_argument("K must be a zero-skew native pinhole model");
}

double radial_domain_radius_squared(const std::array<double, 5> &d) {
  return std::min(first_positive_root({1.0, d[0], d[1], d[4]}),
                  first_positive_root({1.0, 3*d[0], 5*d[1], 7*d[4]}));
}

ProjectionResult project_visible_sensor_points(const std::vector<XYZ> &sensor_points,
                                               const Camera &camera,
                                               const Transform &camera_from_sensor) {
  validate_camera(camera);
  validate_transform(camera_from_sensor);
  const std::size_t pixel_count = static_cast<std::size_t>(camera.width) * camera.height;
  std::vector<double> depth(pixel_count, std::numeric_limits<double>::infinity());
  std::vector<std::size_t> selected(pixel_count, std::numeric_limits<std::size_t>::max());
  std::vector<double> projected_u(sensor_points.size()), projected_v(sensor_points.size());
  const double radial_limit = radial_domain_radius_squared(camera.d);
  ProjectionResult result;
  result.input_count = sensor_points.size();
  for (std::size_t index = 0; index < sensor_points.size(); ++index) {
    const auto &sensor = sensor_points[index];
    if (!finite(sensor)) continue;
    const auto point = transform_point(camera_from_sensor, sensor);
    if (!finite(point) || point.z <= 0) continue;
    ++result.front_count;
    const double x = point.x/point.z, y = point.y/point.z;
    const double r2 = x*x+y*y;
    if (!std::isfinite(r2) || r2 >= radial_limit) { ++result.radial_excluded_count; continue; }
    const auto &d = camera.d;
    const double radial = 1+d[0]*r2+d[1]*r2*r2+d[4]*r2*r2*r2;
    const double xd = x*radial+2*d[2]*x*y+d[3]*(r2+2*x*x);
    const double yd = y*radial+d[2]*(r2+2*y*y)+2*d[3]*x*y;
    const double u = camera.k[0]*xd+camera.k[2], v = camera.k[4]*yd+camera.k[5];
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0 || v < 0 ||
        u >= camera.width || v >= camera.height) continue;
    ++result.in_frame_count;
    const auto ix = std::min(static_cast<std::uint32_t>(std::floor(u+0.5)), camera.width-1);
    const auto iy = std::min(static_cast<std::uint32_t>(std::floor(v+0.5)), camera.height-1);
    const auto pixel = static_cast<std::size_t>(iy)*camera.width+ix;
    if (point.z < depth[pixel]) {
      depth[pixel] = point.z; selected[pixel] = index;
      projected_u[index] = u; projected_v[index] = v;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    if (selected[pixel] == std::numeric_limits<std::size_t>::max()) continue;
    const auto index = selected[pixel];
    result.points.push_back({index, projected_u[index], projected_v[index], depth[pixel],
                             static_cast<std::uint32_t>(pixel % camera.width),
                             static_cast<std::uint32_t>(pixel / camera.width)});
  }
  return result;
}

ColorResult colorize(const std::vector<XYZ> &source, const Camera &camera,
                     const ImageView &image, const Transform &camera_from_sensor,
                     const Transform &sensor_from_lidar) {
  validate_camera(camera);
  validate_transform(camera_from_sensor);
  validate_transform(sensor_from_lidar);
  if (!image.data || image.width != camera.width || image.height != camera.height ||
      image.step < static_cast<std::size_t>(image.width)*3 ||
      image.size != image.step*image.height)
    throw std::invalid_argument("Raw RGB/BGR image layout does not match native calibration");
  if (source.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Source indices exceed UINT32");
  ColorResult result;
  result.input_count = source.size();
  result.points.reserve(source.size());
  std::vector<XYZ> sensor_points;
  sensor_points.reserve(source.size());
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto &point = source[index];
    if (!finite(point) || (point.x == 0 && point.y == 0 && point.z == 0)) continue;
    const auto sensor = transform_point(sensor_from_lidar, point);
    ColoredPoint output;
    output.x = static_cast<float>(sensor.x); output.y = static_cast<float>(sensor.y);
    output.z = static_cast<float>(sensor.z); output.source_index = static_cast<std::uint32_t>(index);
    if (!std::isfinite(output.x) || !std::isfinite(output.y) || !std::isfinite(output.z))
      throw std::invalid_argument("Transformed coordinates cannot be represented as FLOAT32");
    result.points.push_back(output);
    sensor_points.push_back(sensor);
  }
  const auto projection = project_visible_sensor_points(sensor_points, camera, camera_from_sensor);
  result.front_count = projection.front_count;
  result.radial_excluded_count = projection.radial_excluded_count;
  result.in_frame_count = projection.in_frame_count;
  for (const auto &visible : projection.points) {
    const auto row = visible.pixel_y, column = visible.pixel_x;
    const auto *sample = image.data+row*image.step+column*3;
    const auto red = sample[image.bgr ? 2 : 0], green = sample[1], blue = sample[image.bgr ? 0 : 2];
    auto &point = result.points[visible.point_index];
    point.rgb = (static_cast<std::uint32_t>(red)<<16) | (static_cast<std::uint32_t>(green)<<8) | blue;
    point.camera_colored = true;
    ++result.colored_count;
  }
  return result;
}
}  // namespace bev_visualization
