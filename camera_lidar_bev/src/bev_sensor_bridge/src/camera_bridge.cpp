#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "bev_sensor_bridge/camera_support.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

namespace bev_sensor_bridge {
using SteadyClock = std::chrono::steady_clock;
using Status = diagnostic_msgs::msg::DiagnosticStatus;

struct CaptureState {
  std::uint64_t frames{0};
  std::uint64_t failures{0};
  int width{0};
  int height{0};
  bool calibrated{false};
  std::string error;
  SteadyClock::time_point first_frame{};
  SteadyClock::time_point last_frame{};
};

class CameraBridge final : public rclcpp::Node {
 public:
  CameraBridge() : Node("front_camera_bridge"), started_(SteadyClock::now()) {
    source_mode_ = declare_parameter("source_mode", "camera");
    device_ = declare_parameter("device",
        "/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.2:1.0-video-index0");
    width_ = declare_parameter("width", 640);
    height_ = declare_parameter("height", 480);
    fps_ = declare_parameter("fps", 30.0);
    fourcc_ = declare_parameter("fourcc", "MJPG");
    frame_id_ = declare_parameter("frame_id", "camera_front_optical_frame");
    const auto image_topic = declare_parameter("image_topic", "/camera/front/image_raw");
    const auto info_topic = declare_parameter("camera_info_topic", "/camera/front/camera_info");
    const auto calibration_file = declare_parameter("calibration_file", "");
    max_frames_ = declare_parameter("max_frames", 0);
    stale_after_s_ = declare_parameter("stale_after_s", 1.0);
    bool simulation_clock = false;
    get_parameter("use_sim_time", simulation_clock);
    if (simulation_clock) {
      throw std::invalid_argument("Capture uses host receipt stamps: use_sim_time must be false; replay bags with only the sensor probe");
    }
    if (source_mode_ != "camera" && source_mode_ != "synthetic") {
      throw std::invalid_argument("source_mode must be camera or synthetic");
    }
    if (width_ <= 0 || height_ <= 0 || width_ > 8192 || height_ > 8192 ||
        static_cast<std::int64_t>(width_) * height_ > 16777216 ||
        !std::isfinite(fps_) || fps_ <= 0.0 || fps_ > 1000.0 ||
        max_frames_ < 0 || !std::isfinite(stale_after_s_) || stale_after_s_ <= 0.0 ||
        fourcc_.size() != 4 || frame_id_.empty() || image_topic.empty() || info_topic.empty()) {
      throw std::invalid_argument("Invalid capture dimensions, FPS, fourcc, topics, frame, or freshness settings");
    }
    calibration_ = load_camera_calibration(calibration_file);
    if (source_mode_ == "camera") {
      if (device_.empty() || device_.rfind("/dev/", 0) != 0) {
        throw std::invalid_argument("device must be an explicit /dev/video or /dev/v4l path");
      }
      if (!capture_.open(device_, cv::CAP_V4L2)) {
        throw std::runtime_error("Cannot open front camera " + device_ + "; check device availability, permissions, and competing capture processes");
      }
      const int codec = cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
      set_capture_property(cv::CAP_PROP_FOURCC, codec, "fourcc");
      set_capture_property(cv::CAP_PROP_FRAME_WIDTH, width_, "width");
      set_capture_property(cv::CAP_PROP_FRAME_HEIGHT, height_, "height");
      set_capture_property(cv::CAP_PROP_FPS, fps_, "fps");
      // Four buffers match the verified UVC setup; backend buffering contributes
      // unknown capture-to-publication delay and is not exposure synchronization.
      set_capture_property(cv::CAP_PROP_BUFFERSIZE, 4, "buffer count");
      RCLCPP_INFO(get_logger(), "Opened %s; backend reports %.0fx%.0f at %.2f FPS",
          device_.c_str(), capture_.get(cv::CAP_PROP_FRAME_WIDTH),
          capture_.get(cv::CAP_PROP_FRAME_HEIGHT), capture_.get(cv::CAP_PROP_FPS));
    } else {
      RCLCPP_INFO(get_logger(), "Synthetic test images enabled; no camera device opened");
    }
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic, qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(info_topic, qos);
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 5);
    capture_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    diagnostic_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    const auto period = std::chrono::nanoseconds(static_cast<std::int64_t>(1e9 / fps_));
    capture_timer_ = create_wall_timer(period, [this]() { publish_frame(); }, capture_group_);
    diagnostics_timer_ = create_wall_timer(std::chrono::seconds(1),
        [this]() { publish_diagnostics(); }, diagnostic_group_);
    RCLCPP_WARN(get_logger(), "Image headers use host receipt after read; exposure time and sensor synchronization are unverified");
  }

 private:
  void set_capture_property(int property, double value, const char *name) {
    if (!capture_.set(property, value)) {
      RCLCPP_WARN(get_logger(), "Camera backend did not accept requested %s", name);
    }
  }

  void capture_error(const std::string &error) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    ++state_.failures;
    state_.error = error;
  }

  void publish_frame() {
    try {
      cv::Mat image;
      if (source_mode_ == "synthetic") {
        image = cv::Mat(height_, width_, CV_8UC3,
            cv::Scalar(synthetic_sequence_++ % 256, 64, 128));
      } else if (!capture_.read(image) || image.empty()) {
        capture_error("Camera read failed; no image published");
        return;
      }
      // Capture returns a decoded buffer. This is a host receipt timestamp,
      // not the sensor's exposure time, even if ROS header ages appear small.
      std_msgs::msg::Header header;
      header.stamp = host_clock_.now();
      header.frame_id = frame_id_;
      auto message = bgr_image_message(image, header);
      auto info = camera_info_for_frame(calibration_, message.width, message.height, header);
      std::string error = info.error;
      if (image.cols != width_ || image.rows != height_) {
        if (!error.empty()) error += "; ";
        error += "Actual camera dimensions differ from requested mode";
      }
      image_pub_->publish(message);
      info_pub_->publish(info.message);
      std::uint64_t frame_count;
      double mean_rate = 0.0;
      {
        std::lock_guard<std::mutex> guard(state_mutex_);
        const auto receipt = SteadyClock::now();
        if (state_.frames == 0) state_.first_frame = receipt;
        state_.last_frame = receipt;
        ++state_.frames;
        state_.width = image.cols;
        state_.height = image.rows;
        state_.calibrated = info.calibrated;
        state_.error = error;
        frame_count = state_.frames;
        const double interval = std::chrono::duration<double>(receipt - state_.first_frame).count();
        if (interval > 0.0) mean_rate = (state_.frames - 1) / interval;
      }
      if (max_frames_ > 0 && frame_count >= static_cast<std::uint64_t>(max_frames_)) {
        capture_timer_->cancel();
        RCLCPP_INFO(get_logger(), "Published %lu frames; mean receipt rate %.2f Hz",
            static_cast<unsigned long>(frame_count), mean_rate);
        rclcpp::shutdown();
      }
    } catch (const std::exception &error) {
      capture_error(std::string("Capture/conversion failed: ") + error.what());
    }
  }

  static void add_value(Status &status, const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = value;
    status.values.push_back(item);
  }

  void publish_diagnostics() {
    CaptureState snapshot;
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      snapshot = state_;
    }
    const auto steady_now = SteadyClock::now();
    const double age = std::chrono::duration<double>(steady_now -
        (snapshot.frames == 0 ? started_ : snapshot.last_frame)).count();
    Status status;
    status.name = "camera_lidar_bev/front_camera_bridge";
    status.hardware_id = source_mode_ == "camera" ? device_ : "synthetic_test_source";
    status.level = Status::WARN;
    status.message = snapshot.calibrated ?
        "Host receipt stamps; exposure timing and sensor synchronization unverified" :
        "Camera uncalibrated; host receipt stamps and sensor synchronization unverified";
    if (!snapshot.error.empty()) {
      status.level = Status::ERROR;
      status.message = snapshot.error;
    }
    if (age > stale_after_s_) {
      status.level = Status::ERROR;
      status.message = "Camera input stale or unavailable; no fresh image received";
    }
    add_value(status, "source_mode", source_mode_);
    add_value(status, "frames_published", std::to_string(snapshot.frames));
    add_value(status, "read_or_conversion_failures", std::to_string(snapshot.failures));
    add_value(status, "calibrated", snapshot.calibrated ? "true" : "false");
    add_value(status, "timestamp_source", "host_receipt_after_read");
    add_value(status, "exposure_timestamps_available", "false");
    add_value(status, "clock_domain", "host_ros_system_time");
    add_value(status, "synchronization_validated", "false");
    add_value(status, "frame_id", frame_id_);
    add_value(status, "actual_width", std::to_string(snapshot.width));
    add_value(status, "actual_height", std::to_string(snapshot.height));
    add_value(status, "requested_width", std::to_string(width_));
    add_value(status, "requested_height", std::to_string(height_));
    add_value(status, "requested_fps", std::to_string(fps_));
    add_value(status, "requested_fourcc", fourcc_);
    add_value(status, "receipt_age_s", std::to_string(age));
    const double interval = std::chrono::duration<double>(snapshot.last_frame - snapshot.first_frame).count();
    add_value(status, "mean_receipt_hz", std::to_string(interval > 0.0 ? (snapshot.frames - 1) / interval : 0.0));
    diagnostic_msgs::msg::DiagnosticArray report;
    report.header.stamp = host_clock_.now();
    report.status.push_back(status);
    diagnostics_pub_->publish(report);
  }

  std::string source_mode_, device_, fourcc_, frame_id_;
  int width_{}, height_{};
  std::int64_t max_frames_{};
  double fps_{}, stale_after_s_{};
  std::uint64_t synthetic_sequence_{0};
  CameraCalibration calibration_;
  // An explicit system clock preserves the stamp contract even if the node's
  // generic use_sim_time parameter is changed after startup.
  rclcpp::Clock host_clock_{RCL_SYSTEM_TIME};
  cv::VideoCapture capture_;
  SteadyClock::time_point started_;
  CaptureState state_;
  std::mutex state_mutex_;
  rclcpp::CallbackGroup::SharedPtr capture_group_, diagnostic_group_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::TimerBase::SharedPtr capture_timer_, diagnostics_timer_;
};
}  // namespace bev_sensor_bridge

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<bev_sensor_bridge::CameraBridge>();
    // Separate groups allow diagnostics to report staleness during a blocked
    // backend read. Capture callbacks themselves never run concurrently.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("front_camera_bridge"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
