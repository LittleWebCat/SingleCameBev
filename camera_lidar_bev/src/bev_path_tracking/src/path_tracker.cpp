#include "bev_path_tracking/pure_pursuit.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

using bev_path_tracking::Pose;
using bev_path_tracking::Result;
class PathTracker final: public rclcpp::Node {
 public:
  PathTracker():Node("path_tracker") {
    parameters_.calibrated=declare_parameter("vehicle_transform_verified",false);
    parameters_.wheelbase=declare_parameter("wheelbase_m",2.7);
    parameters_.lidar_forward=declare_parameter("lidar_forward_from_rear_axle_m",0.);
    parameters_.lidar_left=declare_parameter("lidar_left_from_rear_axle_m",0.);
    parameters_.sensor_yaw=declare_parameter("sensor_yaw_in_vehicle_rad",0.);
    parameters_.min_lookahead=declare_parameter("min_lookahead_m",3.);
    parameters_.max_lookahead=declare_parameter("max_lookahead_m",12.);
    parameters_.lookahead_gain=declare_parameter("lookahead_time_s",1.5);
    parameters_.max_distance=declare_parameter("max_path_distance_m",5.);
    parameters_.max_heading=declare_parameter("max_heading_error_rad",1.0471975512);
    parameters_.max_gap=declare_parameter("max_reference_gap_s",.5);
    parameters_.max_steering=declare_parameter("max_steering_rad",.5);
    parameters_.max_steering_rate=declare_parameter("max_steering_rate_rad_s",.3);
    parameters_.max_speed=declare_parameter("preview_speed_mps",2.);
    parameters_.deceleration=declare_parameter("preview_deceleration_mps2",1.);
    parameters_.lateral_acceleration=declare_parameter("preview_lateral_acceleration_mps2",1.);
    parameters_.stop_margin=declare_parameter("stop_margin_m",.5);
    parameters_.motion_window=declare_parameter("motion_curvature_window_s",.5);
    parameters_.motion_min_speed=declare_parameter("motion_curvature_min_speed_mps",.5);
    parameters_.motion_min_distance=declare_parameter("motion_curvature_min_distance_m",.25);
    maximum_age_=declare_parameter("maximum_pose_age_s",.5);
    source_mode_=declare_parameter("source_mode",std::string("viewer_paired_pose"));
    if (source_mode_!="viewer_paired_pose"&&source_mode_!="live_localization"&&source_mode_!="saved_recording_pose")
      throw std::runtime_error("unsupported pose source mode");
    if (!std::isfinite(maximum_age_)||maximum_age_<=0) throw std::runtime_error("invalid pose age limit");
    const auto csv=declare_parameter("reference_trajectory",std::string(""));
    tracker_=std::make_unique<bev_path_tracking::Tracker>(bev_path_tracking::read_csv(csv),parameters_);
    publisher_=create_publisher<std_msgs::msg::String>("/camera_lidar_bev/path_tracking/status",10);
    subscription_=create_subscription<nav_msgs::msg::Odometry>(
      "/camera_lidar_bev/path_tracking/source_pose",10,
      [this](const nav_msgs::msg::Odometry::SharedPtr message) { receive(*message); });
    timer_=create_wall_timer(std::chrono::milliseconds(100),[this] {
      check_clock();
      if (last_stamp_==0) { publish(Result{},0); return; }
      const double age=(now().nanoseconds()-last_stamp_)*1e-9;
      const double wall_age=std::chrono::duration<double>(std::chrono::steady_clock::now()-received_).count();
      if (age>maximum_age_||age<-.05||wall_age>maximum_age_) {
        tracker_->reset(); Result invalid; invalid.state="localization_stale";
        publish(invalid,last_stamp_);
      }
    });
    RCLCPP_INFO(get_logger(),"Loaded %zu route vertices. Diagnostic outputs only; vehicle transform %s.",
                tracker_->path().size(),parameters_.calibrated?"configured":"UNSET");
  }
 private:
  void check_clock() {
    const auto stamp=now().nanoseconds();
    if (clock_stamp_>0&&stamp<clock_stamp_) { tracker_->reset(); last_stamp_=0; }
    clock_stamp_=stamp;
  }
  void receive(const nav_msgs::msg::Odometry &message) {
    check_clock();
    auto invalid=[this](const char *state) {
      tracker_->reset(); Result result; result.state=state; publish(result,last_stamp_);
    };
    const auto stamp=rclcpp::Time(message.header.stamp).nanoseconds();
    if (message.header.frame_id!="enu"||message.child_frame_id!="os_sensor"||stamp<=0) {
      invalid("invalid_pose_frame_or_stamp"); return;
    }
    if (last_stamp_>0&&stamp<=last_stamp_) { invalid("out_of_order_pose"); return; }
    const double age=(now().nanoseconds()-stamp)*1e-9;
    if (age>maximum_age_||age<-.05) { invalid("localization_stale"); return; }
    const auto &p=message.pose.pose;
    double norm=p.orientation.x*p.orientation.x+p.orientation.y*p.orientation.y+
                p.orientation.z*p.orientation.z+p.orientation.w*p.orientation.w;
    if (!std::isfinite(norm)||norm<1e-12||!std::isfinite(p.position.x)||
        !std::isfinite(p.position.y)||!std::isfinite(p.position.z)) {
      invalid("invalid_pose"); return;
    }
    const double qx=p.orientation.x/std::sqrt(norm),qy=p.orientation.y/std::sqrt(norm);
    const double qz=p.orientation.z/std::sqrt(norm),qw=p.orientation.w/std::sqrt(norm);
    Pose pose{p.position.x,p.position.y,p.position.z,
              std::atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)),stamp*1e-9};
    last_stamp_=stamp; received_=std::chrono::steady_clock::now();
    const auto start=std::chrono::steady_clock::now();
    const auto result=tracker_->update(pose);
    processing_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    publish(result,stamp);
  }
  void publish(const Result &r,std::int64_t stamp) {
    std::ostringstream out;
    out<<std::setprecision(12)<<std::boolalpha;
    auto xyz=[&out](const Pose &p) { out<<"["<<p.x<<","<<p.y<<","<<p.z<<"]"; };
    out<<"{\"frame\":\"enu\",\"source_stamp_ns\":"<<stamp
       <<",\"source_mode\":\""<<source_mode_<<"\",\"diagnostic_only\":true"
       <<",\"tracking_valid\":"<<r.valid<<",\"targets_valid\":"<<r.targets_valid
       <<",\"vehicle_transform_verified\":"<<parameters_.calibrated
       <<",\"pose_reference\":\""<<(parameters_.calibrated?"rear_axle":"os_sensor")<<"\""
       <<",\"state\":\""<<r.state<<"\",\"processing_ms\":"<<processing_ms_
       <<",\"lateral_error_m\":"<<(r.valid?std::to_string(r.lateral_error):"null")
       <<",\"heading_error_rad\":"<<(r.valid?std::to_string(r.heading_error):"null")
       <<",\"progress_m\":"<<r.progress<<",\"remaining_m\":"<<r.remaining
       <<",\"estimated_speed_mps\":"<<(r.speed_valid?std::to_string(r.speed):"null")
       <<",\"steering_rad\":"<<(r.targets_valid?std::to_string(r.steering):"null")
       <<",\"target_speed_mps\":"<<(r.targets_valid?std::to_string(r.target_speed):"null")
       <<",\"curvature_per_m\":"<<(r.targets_valid?std::to_string(r.curvature):"null")
       <<",\"vehicle_curvature_valid\":"<<r.motion_valid
       <<",\"vehicle_curvature_source\":\"localization_pose_estimate\""
       <<",\"motion_curvature_window_s\":"<<parameters_.motion_window
       <<",\"vehicle_curvature_per_m\":"<<(r.motion_valid?std::to_string(r.vehicle_curvature):"null")
       <<",\"motion_speed_mps\":"<<(r.motion_valid?std::to_string(r.motion_speed):"null")
       <<",\"estimated_yaw_rate_rad_s\":"<<(r.motion_valid?std::to_string(r.yaw_rate):"null")
       <<",\"motion_window_span_s\":"<<(r.motion_valid?std::to_string(r.motion_span):"null")
       <<",\"steering_saturated\":"<<r.saturated
       <<",\"lookahead_m\":"<<r.lookahead<<",\"reference_xyz\":[";
    for (std::size_t i=0;i<r.horizon.size();++i) { if(i) out<<","; xyz(r.horizon[i]); }
    out<<"],\"lookahead_xyz\":";
    if(r.valid) xyz(r.target); else out<<"null";
    out<<",\"projection_xyz\":";
    if(r.valid) xyz(r.projection); else out<<"null";
    out<<",\"ego_xyz\":";
    if(r.valid) xyz(r.ego); else out<<"null";
    out<<"}";
    std_msgs::msg::String message; message.data=out.str(); publisher_->publish(message);
  }
  bev_path_tracking::Parameters parameters_;
  std::unique_ptr<bev_path_tracking::Tracker> tracker_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::chrono::steady_clock::time_point received_;
  std::int64_t last_stamp_=0,clock_stamp_=0;
  double maximum_age_=.5,processing_ms_=0;
  std::string source_mode_;
};
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try { rclcpp::spin(std::make_shared<PathTracker>()); }
  catch(const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("path_tracker"),"%s",error.what());
    rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
