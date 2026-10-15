#include "bev_sensor_bridge/camera_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <sensor_msgs/image_encodings.hpp>
#include <yaml-cpp/yaml.h>

namespace bev_sensor_bridge
{
namespace
{

std::uint32_t positive_dimension(const YAML::Node & node, const std::string & name)
{
  if (!node || !node.IsScalar()) {
    throw std::runtime_error(name + " must be a positive integer");
  }
  const double value = node.as<double>();
  if (!std::isfinite(value) || value < 1.0 || std::floor(value) != value ||
    value > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
  {
    throw std::runtime_error(name + " must be a positive uint32 integer");
  }
  return static_cast<std::uint32_t>(value);
}

std::vector<double> matrix(
  const YAML::Node & root, const std::string & name,
  std::uint32_t rows, std::uint32_t columns)
{
  const auto node = root[name];
  if (!node || !node.IsMap()) {
    throw std::runtime_error("missing calibration matrix: " + name);
  }
  if (positive_dimension(node["rows"], name + ".rows") != rows ||
    positive_dimension(node["cols"], name + ".cols") != columns)
  {
    throw std::runtime_error(name + " has incorrect rows/cols");
  }
  const auto data = node["data"];
  const std::size_t expected = static_cast<std::size_t>(rows) * columns;
  if (!data || !data.IsSequence() || data.size() != expected) {
    throw std::runtime_error(name + ".data has incorrect length");
  }
  std::vector<double> values;
  values.reserve(expected);
  for (const auto & element : data) {
    if (!element.IsScalar()) {
      throw std::runtime_error(name + ".data must contain numeric scalars");
    }
    const double value = element.as<double>();
    if (!std::isfinite(value)) {
      throw std::runtime_error(name + ".data must contain only finite values");
    }
    values.push_back(value);
  }
  return values;
}

bool near(double value, double expected, double tolerance = 1e-9)
{
  return std::abs(value - expected) <= tolerance;
}

void validate_intrinsics(const std::array<double, 9> & k)
{
  if (k[0] <= 0.0 || k[4] <= 0.0 || !near(k[1], 0.0) || !near(k[3], 0.0) ||
    !near(k[6], 0.0) || !near(k[7], 0.0) || !near(k[8], 1.0))
  {
    throw std::runtime_error("camera_matrix must contain positive focal lengths and ROS pinhole form");
  }
}

void validate_rectification(const std::array<double, 9> & r)
{
  // R is a rotation, not an all-zero placeholder or arbitrary 3x3 transform.
  // Calibration exporters may round coefficients, hence this tolerance.
  constexpr double tolerance = 1e-3;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t other = 0; other < 3; ++other) {
      double product = 0.0;
      for (std::size_t column = 0; column < 3; ++column) {
        product += r[row * 3 + column] * r[other * 3 + column];
      }
      if (!near(product, row == other ? 1.0 : 0.0, tolerance)) {
        throw std::runtime_error("rectification_matrix must be an orthonormal rotation");
      }
    }
  }
  const double determinant =
    r[0] * (r[4] * r[8] - r[5] * r[7]) -
    r[1] * (r[3] * r[8] - r[5] * r[6]) +
    r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (!near(determinant, 1.0, tolerance)) {
    throw std::runtime_error("rectification_matrix must have determinant +1");
  }
}

void validate_projection(const std::array<double, 12> & p)
{
  if (p[0] <= 0.0 || p[5] <= 0.0 || !near(p[1], 0.0) || !near(p[4], 0.0) ||
    !near(p[8], 0.0) || !near(p[9], 0.0) || !near(p[10], 1.0) || !near(p[11], 0.0))
  {
    throw std::runtime_error("projection_matrix must contain positive focal lengths and ROS pinhole form");
  }
}

}  // namespace

CameraCalibration load_camera_calibration(const std::string & path)
{
  CameraCalibration calibration;
  if (path.empty()) {
    return calibration;
  }
  try {
    const auto root = YAML::LoadFile(path);
    if (!root.IsMap()) {
      throw std::runtime_error("calibration YAML root must be a mapping");
    }
    auto & info = calibration.message;
    info.width = positive_dimension(root["image_width"], "image_width");
    info.height = positive_dimension(root["image_height"], "image_height");
    const auto model_node = root["distortion_model"];
    if (!model_node || !model_node.IsScalar()) {
      throw std::runtime_error("missing distortion_model");
    }
    info.distortion_model = model_node.as<std::string>();
    std::uint32_t distortion_count = 0;
    if (info.distortion_model == "plumb_bob") {
      distortion_count = 5;
    } else if (info.distortion_model == "rational_polynomial") {
      distortion_count = 8;
    } else if (info.distortion_model == "equidistant") {
      distortion_count = 4;
    } else {
      throw std::runtime_error("unsupported distortion_model: " + info.distortion_model);
    }
    const auto k = matrix(root, "camera_matrix", 3, 3);
    const auto r = matrix(root, "rectification_matrix", 3, 3);
    const auto p = matrix(root, "projection_matrix", 3, 4);
    info.d = matrix(root, "distortion_coefficients", 1, distortion_count);
    std::copy(k.begin(), k.end(), info.k.begin());
    std::copy(r.begin(), r.end(), info.r.begin());
    std::copy(p.begin(), p.end(), info.p.begin());
    validate_intrinsics(info.k);
    validate_rectification(info.r);
    validate_projection(info.p);
    calibration.calibrated = true;
    return calibration;
  } catch (const std::exception & error) {
    throw std::runtime_error("invalid camera calibration '" + path + "': " + error.what());
  }
}

CameraInfoResult camera_info_for_frame(
  const CameraCalibration & calibration, std::uint32_t width,
  std::uint32_t height, const std_msgs::msg::Header & header)
{
  if (width == 0 || height == 0) {
    throw std::invalid_argument("capture width and height must be positive");
  }
  CameraInfoResult result;
  if (calibration.calibrated) {
    if (calibration.message.width == width && calibration.message.height == height) {
      result.message = calibration.message;
      result.calibrated = true;
    } else {
      result.error = "calibration resolution " + std::to_string(calibration.message.width) + "x" +
        std::to_string(calibration.message.height) + " differs from capture " +
        std::to_string(width) + "x" + std::to_string(height) +
        "; publishing uncalibrated CameraInfo";
    }
  }
  result.message.width = width;
  result.message.height = height;
  result.message.header = header;
  return result;
}

sensor_msgs::msg::Image bgr_image_message(
  const cv::Mat & frame, const std_msgs::msg::Header & header)
{
  if (frame.empty() || frame.dims != 2 || frame.type() != CV_8UC3) {
    throw std::invalid_argument("camera frame must be a nonempty two-dimensional CV_8UC3 image");
  }
  const std::size_t row_bytes = static_cast<std::size_t>(frame.cols) * 3U;
  if (row_bytes > std::numeric_limits<std::uint32_t>::max() ||
    static_cast<std::size_t>(frame.rows) > std::numeric_limits<std::size_t>::max() / row_bytes)
  {
    throw std::invalid_argument("camera frame dimensions exceed ROS Image storage limits");
  }
  sensor_msgs::msg::Image message;
  message.header = header;
  message.width = static_cast<std::uint32_t>(frame.cols);
  message.height = static_cast<std::uint32_t>(frame.rows);
  message.encoding = sensor_msgs::image_encodings::BGR8;
  message.is_bigendian = 0;
  message.step = static_cast<std::uint32_t>(row_bytes);
  message.data.resize(row_bytes * static_cast<std::size_t>(frame.rows));
  for (int row = 0; row < frame.rows; ++row) {
    std::memcpy(message.data.data() + static_cast<std::size_t>(row) * row_bytes,
      frame.ptr(row), row_bytes);
  }
  return message;
}

}  // namespace bev_sensor_bridge
