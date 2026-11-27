#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"

namespace bev_sensor_bridge {
using SteadyClock = std::chrono::steady_clock;
using Status = diagnostic_msgs::msg::DiagnosticStatus;

struct InputState {
  std::string name;
  std::string topic;
  std::string frame;
  std::uint64_t count{0};
  SteadyClock::time_point first_receipt{};
  SteadyClock::time_point last_receipt{};
  std::int64_t header_ns{0};
};

class SensorProbe final : public rclcpp::Node {
 public:
  SensorProbe() : Node("sensor_probe") {
    stale_after_s_ = declare_parameter("stale_after_s", 1.0);
    future_tolerance_s_ = declare_parameter("future_tolerance_s", 0.05);
    if (!std::isfinite(stale_after_s_) || stale_after_s_ <= 0.0 ||
        !std::isfinite(future_tolerance_s_) || future_tolerance_s_ < 0.0) {
      throw std::invalid_argument("Freshness thresholds must be finite and valid");
    }
    lidar_ = make_input("lidar", "lidar_topic", "/points");
    camera_ = make_input("front_camera", "camera_topic", "/camera/front/image_raw");
    camera_info_ = make_input("camera_info", "camera_info_topic", "/camera/front/camera_info");
    imu_ = make_input("imu", "imu_topic", "/sensing/imu/imu_data");
    odom_ = make_input("odometry", "odom_topic", "/lio_sam/mapping/odometry");
    const auto lidar_reliability = declare_parameter("lidar_reliability", "reliable");
    if (lidar_reliability != "reliable" && lidar_reliability != "best_effort") {
      throw std::invalid_argument("lidar_reliability must be reliable or best_effort");
    }

    const auto qos = rclcpp::SensorDataQoS().keep_last(5);
    auto lidar_qos = rclcpp::SensorDataQoS().keep_last(1);
    if (lidar_reliability == "reliable") lidar_qos.reliable();
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_.topic, lidar_qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          observe(lidar_, msg->header);
          point_fields_.clear();
          has_point_time_ = false;
          bool has_x = false, has_y = false, has_z = false;
          for (const auto &field : msg->fields) {
            if (!point_fields_.empty()) point_fields_ += ",";
            point_fields_ += field.name;
            has_x |= field.name == "x";
            has_y |= field.name == "y";
            has_z |= field.name == "z";
            has_point_time_ |= field.name == "t" || field.name == "time" ||
                               field.name == "timestamp" || field.name == "offset_time";
          }
          has_xyz_ = has_x && has_y && has_z;
          point_count_ = static_cast<std::uint64_t>(msg->width) * msg->height;
          const auto minimum_row_bytes = static_cast<std::uint64_t>(msg->width) * msg->point_step;
          const auto expected_data_bytes = static_cast<std::uint64_t>(msg->height) * msg->row_step;
          valid_payload_layout_ = msg->point_step > 0 &&
              msg->row_step >= minimum_row_bytes && msg->data.size() == expected_data_bytes;
        });
    camera_sub_ = create_subscription<sensor_msgs::msg::Image>(
        camera_.topic, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          observe(camera_, msg->header);
        });
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_.topic, qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
          observe(camera_info_, msg->header);
          has_intrinsics_ = msg->width > 0 && msg->height > 0 &&
              std::isfinite(msg->k[0]) && std::isfinite(msg->k[4]) &&
              msg->k[0] > 0.0 && msg->k[4] > 0.0;
        });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_.topic, qos, [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
          observe(imu_, msg->header);
        });
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_.topic, qos, [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
          observe(odom_, msg->header);
        });
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", 5);
    timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_diagnostics(); });
    RCLCPP_INFO(get_logger(), "Input probe ready; this node publishes diagnostics only");
  }

 private:
  InputState make_input(const std::string &name, const std::string &parameter,
                        const std::string &default_topic) {
    InputState input;
    input.name = name;
    input.topic = declare_parameter(parameter, default_topic);
    if (input.topic.empty()) throw std::invalid_argument(parameter + " must not be empty");
    return input;
  }

  void observe(InputState &input, const std_msgs::msg::Header &header) {
    const auto receipt = SteadyClock::now();
    if (input.count == 0) input.first_receipt = receipt;
    input.last_receipt = receipt;
    ++input.count;
    input.frame = header.frame_id;
    input.header_ns = static_cast<std::int64_t>(header.stamp.sec) * 1000000000LL +
                      header.stamp.nanosec;
  }

  static void add_value(Status &status, const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue entry;
    entry.key = key;
    entry.value = value;
    status.values.push_back(entry);
  }

  static void flag(Status &status, std::uint8_t level, const std::string &message) {
    if (level >= status.level) {
      status.level = level;
      status.message = message;
    }
  }

  Status input_status(const InputState &input, const SteadyClock::time_point receipt_now,
                      const std::int64_t ros_now_ns) const {
    Status status;
    status.name = "camera_lidar_bev/" + input.name;
    status.hardware_id = input.topic;
    status.level = Status::OK;
    status.message = "Receiving messages; arrival/header checks do not establish synchronization";
    add_value(status, "topic", input.topic);
    add_value(status, "messages_received", std::to_string(input.count));
    add_value(status, "frame_id", input.frame);
    if (input.count == 0) {
      flag(status, Status::WARN, "Waiting for input; verify topic name, publisher and QoS");
      return status;
    }
    const double receipt_age = std::chrono::duration<double>(receipt_now - input.last_receipt).count();
    const double sample_interval = std::chrono::duration<double>(input.last_receipt - input.first_receipt).count();
    const double mean_hz = sample_interval > 0.0 ? (input.count - 1) / sample_interval : 0.0;
    add_value(status, "receipt_age_s", std::to_string(receipt_age));
    add_value(status, "mean_receipt_hz", std::to_string(mean_hz));
    if (receipt_age > stale_after_s_) flag(status, Status::ERROR, "Input stopped or became stale");
    if (input.frame.empty()) flag(status, Status::WARN, "Missing frame_id");
    if (input.header_ns <= 0 || ros_now_ns <= 0) {
      flag(status, Status::WARN, "Header timestamp or ROS clock is zero/invalid");
    } else {
      const double header_age = static_cast<double>(ros_now_ns - input.header_ns) / 1e9;
      add_value(status, "header_age_s_assuming_common_clock", std::to_string(header_age));
      if (header_age < -future_tolerance_s_) {
        flag(status, Status::WARN, "Header timestamp is in the future; verify clock domains");
      } else if (header_age > stale_after_s_) {
        flag(status, Status::WARN, "Header timestamp is old; verify latency and clock domains");
      }
    }
    return status;
  }

  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray array;
    const auto ros_now = now();
    array.header.stamp = ros_now;
    const auto receipt_now = SteadyClock::now();
    auto lidar_status = input_status(lidar_, receipt_now, ros_now.nanoseconds());
    if (lidar_.count > 0) {
      add_value(lidar_status, "point_fields", point_fields_);
      add_value(lidar_status, "declared_point_count", std::to_string(point_count_));
      add_value(lidar_status, "payload_layout_valid", valid_payload_layout_ ? "true" : "false");
      add_value(lidar_status, "point_time_field_present", has_point_time_ ? "true" : "false");
      add_value(lidar_status, "point_time_units_and_origin", "unverified; field presence alone is insufficient for deskew");
      if (!has_xyz_ || point_count_ == 0) flag(lidar_status, Status::ERROR, "Point cloud lacks XYZ fields or contains no points");
      if (!valid_payload_layout_) flag(lidar_status, Status::ERROR, "Point cloud payload size/stride is inconsistent");
      if (!has_point_time_) flag(lidar_status, Status::WARN, "Per-point timing field missing; deskew prerequisite unmet");
    }
    array.status.push_back(lidar_status);
    array.status.push_back(input_status(camera_, receipt_now, ros_now.nanoseconds()));
    auto info_status = input_status(camera_info_, receipt_now, ros_now.nanoseconds());
    if (camera_info_.count > 0) {
      add_value(info_status, "positive_focal_lengths_present", has_intrinsics_ ? "true" : "false");
      if (!has_intrinsics_) flag(info_status, Status::WARN, "Camera intrinsics missing/invalid");
    }
    array.status.push_back(info_status);
    array.status.push_back(input_status(imu_, receipt_now, ros_now.nanoseconds()));
    array.status.push_back(input_status(odom_, receipt_now, ros_now.nanoseconds()));
    Status readiness;
    readiness.name = "camera_lidar_bev/calibration_and_fusion";
    readiness.hardware_id = "project_foundation";
    readiness.level = Status::WARN;
    readiness.message = "Calibration, motion validation and BEV perception are pending";
    add_value(readiness, "sensor_clock_alignment", "unverified");
    add_value(readiness, "extrinsic_calibration", "unverified");
    add_value(readiness, "occupancy_pipeline", "not implemented in this milestone");
    array.status.push_back(readiness);
    diagnostics_pub_->publish(array);
  }

  double stale_after_s_{};
  double future_tolerance_s_{};
  InputState lidar_, camera_, camera_info_, imu_, odom_;
  bool has_point_time_{false}, has_xyz_{false}, has_intrinsics_{false};
  bool valid_payload_layout_{false};
  std::uint64_t point_count_{0};
  std::string point_fields_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace bev_sensor_bridge

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<bev_sensor_bridge::SensorProbe>());
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("sensor_probe"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
