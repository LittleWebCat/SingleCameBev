#ifndef BEV_SENSOR_BRIDGE__CAMERA_SUPPORT_HPP_
#define BEV_SENSOR_BRIDGE__CAMERA_SUPPORT_HPP_

#include <cstdint>
#include <string>

#include <opencv2/core/mat.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace bev_sensor_bridge
{

struct CameraCalibration
{
  sensor_msgs::msg::CameraInfo message;
  bool calibrated{false};
};

struct CameraInfoResult
{
  sensor_msgs::msg::CameraInfo message;
  bool calibrated{false};
  std::string error;
};

// Empty path intentionally selects an uncalibrated camera. A supplied file must
// contain the standard ROS calibration matrices K, D, R, and P; invalid files
// throw std::runtime_error. Intrinsics and rectification are never guessed.
CameraCalibration load_camera_calibration(const std::string & path);

// Uses the caller's acquisition header and actual capture dimensions. A
// calibration-resolution mismatch returns zeroed, uncalibrated parameters plus
// an error; it never scales intrinsics. Zero capture dimensions throw.
CameraInfoResult camera_info_for_frame(
  const CameraCalibration & calibration, std::uint32_t width,
  std::uint32_t height, const std_msgs::msg::Header & header);

// Copies a two-dimensional CV_8UC3 image into tightly packed ROS BGR8 storage.
// Supports non-contiguous OpenCV ROIs and shares the caller's acquisition header
// with CameraInfo. Empty images or incompatible types throw std::invalid_argument.
sensor_msgs::msg::Image bgr_image_message(
  const cv::Mat & frame, const std_msgs::msg::Header & header);

}  // namespace bev_sensor_bridge

#endif  // BEV_SENSOR_BRIDGE__CAMERA_SUPPORT_HPP_
