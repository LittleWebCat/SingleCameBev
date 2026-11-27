#pragma once
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace bev_path_tracking {
struct Pose { double x=0, y=0, z=0, yaw=0, stamp=0; };
struct Point : Pose { double s=0; int part=0; };
struct Parameters {
  bool calibrated=false;
  double wheelbase=2.7, lidar_forward=0, lidar_left=0, sensor_yaw=0;
  double min_lookahead=3, max_lookahead=12, lookahead_gain=1.5;
  double max_distance=5, max_heading=1.0471975512, max_gap=.5;
  double max_steering=.5, max_steering_rate=.3, max_speed=2;
  double deceleration=1, lateral_acceleration=1, stop_margin=.5;
  double motion_window=.5, motion_min_speed=.5, motion_min_distance=.25;
};
struct Result {
  bool valid=false, targets_valid=false, speed_valid=false, saturated=false, finished=false;
  bool motion_valid=false;
  std::string state="waiting";
  double lateral_error=0, heading_error=0, progress=0, remaining=0;
  double lookahead=0, speed=0, curvature=0, steering=0, target_speed=0;
  double vehicle_curvature=0, motion_speed=0, yaw_rate=0, motion_span=0;
  Pose ego, projection, target;
  std::vector<Pose> horizon;
};
double wrap(double angle);
Pose vehicle_pose(const Pose &sensor, const Parameters &parameters);
std::vector<Pose> read_csv(const std::string &filename);
class Tracker {
 public:
  Tracker(std::vector<Pose> source, Parameters parameters);
  Result update(const Pose &sensor);
  void reset();
  const std::vector<Point> &path() const { return points_; }
 private:
  Pose at(double s, int part) const;
  Parameters p_;
  std::vector<Point> points_;
  bool acquired_=false, previous_=false, steering_previous_=false;
  int part_=0;
  double progress_=0, speed_=0, steering_=0;
  Pose last_;
  std::deque<Pose> motion_;
};
}
