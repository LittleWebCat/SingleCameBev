#include "bev_path_tracking/pure_pursuit.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace bev_path_tracking {
namespace {
constexpr double pi=3.14159265358979323846;
bool finite(const Pose &p) {
  return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&
         std::isfinite(p.yaw)&&std::isfinite(p.stamp);
}
Pose interpolate(const Pose &a, const Pose &b, double f) {
  return {a.x+f*(b.x-a.x), a.y+f*(b.y-a.y), a.z+f*(b.z-a.z),
          wrap(a.yaw+f*wrap(b.yaw-a.yaw)), a.stamp+f*(b.stamp-a.stamp)};
}
double distance(const Pose &a, const Pose &b) { return std::hypot(a.x-b.x,a.y-b.y); }
}
double wrap(double angle) { return std::remainder(angle, 2*pi); }
Pose vehicle_pose(const Pose &sensor, const Parameters &p) {
  if (!p.calibrated) return sensor;
  Pose rear=sensor;
  rear.yaw=wrap(sensor.yaw-p.sensor_yaw);
  rear.x-=std::cos(rear.yaw)*p.lidar_forward-std::sin(rear.yaw)*p.lidar_left;
  rear.y-=std::sin(rear.yaw)*p.lidar_forward+std::cos(rear.yaw)*p.lidar_left;
  return rear;
}
std::vector<Pose> read_csv(const std::string &filename) {
  std::ifstream stream(filename);
  if (!stream) throw std::runtime_error("cannot open reference trajectory");
  std::string line;
  std::getline(stream,line);
  if (!line.empty()&&line.back()=='\r') line.pop_back();
  if (line!="stamp_sec,x_m,y_m,z_m,qx,qy,qz,qw")
    throw std::runtime_error("reference trajectory columns differ from mapping export");
  std::vector<Pose> result;
  while (std::getline(stream,line)) {
    if (!line.empty()&&line.back()=='\r') line.pop_back();
    std::stringstream row(line);
    std::vector<double> values;
    std::string value;
    while (std::getline(row,value,',')) {
      std::size_t used=0;
      double number=std::stod(value,&used);
      if (used!=value.size()||!std::isfinite(number)) throw std::runtime_error("invalid reference value");
      values.push_back(number);
    }
    if (values.size()!=8) throw std::runtime_error("incomplete reference row");
    double norm=0;
    for (std::size_t i=4;i<8;++i) norm+=values[i]*values[i];
    if (norm<1e-12||values[0]<=0) throw std::runtime_error("invalid reference quaternion/timestamp");
    double x=values[4]/std::sqrt(norm), y=values[5]/std::sqrt(norm);
    double z=values[6]/std::sqrt(norm), w=values[7]/std::sqrt(norm);
    result.push_back({values[1],values[2],values[3],std::atan2(2*(w*z+x*y),1-2*(y*y+z*z)),values[0]});
  }
  return result;
}
Tracker::Tracker(std::vector<Pose> source, Parameters parameters):p_(parameters) {
  for (double value : {p_.wheelbase,p_.min_lookahead,p_.max_lookahead,p_.max_distance,p_.max_heading,
        p_.max_gap,p_.max_steering,p_.max_steering_rate,p_.max_speed,p_.deceleration,p_.lateral_acceleration,
        p_.motion_window,p_.motion_min_speed,p_.motion_min_distance}) {
    if (!std::isfinite(value)||value<=0) throw std::runtime_error("tracking limits must be positive and finite");
  }
  for (double value : {p_.lidar_forward,p_.lidar_left,p_.sensor_yaw,p_.lookahead_gain,p_.stop_margin})
    if (!std::isfinite(value)) throw std::runtime_error("invalid vehicle/tracking parameter");
  if (p_.max_lookahead<p_.min_lookahead||p_.lookahead_gain<0||p_.stop_margin<0||p_.max_heading>pi)
    throw std::runtime_error("inconsistent tracking limits");
  double previous_stamp=0;
  int part=0;
  for (const auto &sensor:source) {
    if (!finite(sensor)||sensor.stamp<=previous_stamp) throw std::runtime_error("reference times must increase");
    if (previous_stamp>0&&sensor.stamp-previous_stamp>p_.max_gap) ++part;
    previous_stamp=sensor.stamp;
    const Pose pose=vehicle_pose(sensor,p_);
    if (!points_.empty()&&points_.back().part==part&&distance(pose,points_.back())<.001) continue;
    Point point;
    static_cast<Pose&>(point)=pose;
    point.part=part;
    point.s=points_.empty()?0:points_.back().s+distance(point,points_.back());
    points_.push_back(point);
  }
  if (points_.size()<2) throw std::runtime_error("reference has no usable motion");
  bool usable=false;
  for (std::size_t i=1;i<points_.size();++i)
    usable|=points_[i].part==points_[i-1].part&&points_[i].s>points_[i-1].s;
  if (!usable) throw std::runtime_error("reference has no continuous segment");
}
void Tracker::reset() {
  acquired_=previous_=steering_previous_=false;
  speed_=steering_=progress_=0;
  motion_.clear();
}
Pose Tracker::at(double s,int part) const {
  for (std::size_t i=1;i<points_.size();++i) {
    const auto &a=points_[i-1], &b=points_[i];
    if (a.part!=part||b.part!=part||b.s<a.s+1e-9) continue;
    if (s<=b.s) return interpolate(a,b,std::clamp((s-a.s)/(b.s-a.s),0.,1.));
  }
  for (auto it=points_.rbegin();it!=points_.rend();++it) if (it->part==part) return *it;
  throw std::runtime_error("missing reference segment");
}
Result Tracker::update(const Pose &sensor) {
  Result r;
  if (!finite(sensor)||sensor.stamp<=0) { reset(); r.state="invalid_pose"; return r; }
  r.ego=vehicle_pose(sensor,p_);
  double dt=previous_?r.ego.stamp-last_.stamp:0;
  if (previous_&&(dt<=0||dt>.5)) { reset(); dt=0; }
  if (previous_) {
    double movement=distance(r.ego,last_);
    if (movement/dt>40) { reset(); r.state="pose_jump"; return r; }
    if (p_.calibrated&&movement>.05&&
        ((r.ego.x-last_.x)*std::cos(r.ego.yaw)+(r.ego.y-last_.y)*std::sin(r.ego.yaw))<-.05) {
      reset(); r.state="reverse_motion_unsupported"; return r;
    }
    speed_=previous_?.7*speed_+.3*movement/dt:movement/dt;
    r.speed_valid=true;
  }
  // Curvature from observed rear-axle motion, independent of desired targets.
  // This estimates yaw-rate / path speed under the planar no-slip assumption.
  bool moving_forward=false;
  if (previous_) {
    const double dx=r.ego.x-last_.x,dy=r.ego.y-last_.y;
    moving_forward=(dx*std::cos(r.ego.yaw)+dy*std::sin(r.ego.yaw))>0&&
                   distance(r.ego,last_)/dt>=p_.motion_min_speed;
    if (!moving_forward||std::abs(wrap(r.ego.yaw-last_.yaw))/dt>2.) motion_.clear();
  }
  if (p_.calibrated) {
    motion_.push_back(r.ego);
    while (motion_.size()>256||(!motion_.empty()&&r.ego.stamp-motion_.front().stamp>p_.motion_window+1e-6))
      motion_.pop_front();
    if (moving_forward&&motion_.size()>=2) {
      double travel=0,heading=0;
      for (std::size_t i=1;i<motion_.size();++i) {
        travel+=distance(motion_[i],motion_[i-1]);
        heading+=wrap(motion_[i].yaw-motion_[i-1].yaw);
      }
      r.motion_span=r.ego.stamp-motion_.front().stamp;
      if (r.motion_span>=.8*p_.motion_window-1e-6&&travel>=p_.motion_min_distance&&
          travel/r.motion_span>=p_.motion_min_speed) {
        r.motion_valid=true;
        r.motion_speed=travel/r.motion_span;
        r.yaw_rate=heading/r.motion_span;
        r.vehicle_curvature=heading/travel;
      }
    }
  }
  previous_=true; last_=r.ego;
  r.speed=speed_;
  double best=std::numeric_limits<double>::infinity();
  double best_s=0;
  int best_part=-1;
  Pose projection;
  double lateral=0;
  for (std::size_t i=1;i<points_.size();++i) {
    const auto &a=points_[i-1], &b=points_[i];
    if (a.part!=b.part||b.s-a.s<1e-9) continue;
    if (acquired_&&(a.part!=part_||b.s<progress_-2||a.s>progress_+10)) continue;
    double dx=b.x-a.x,dy=b.y-a.y,length=b.s-a.s;
    if (p_.calibrated&&std::cos(std::atan2(dy,dx)-a.yaw)<.2) continue;
    double f=std::clamp(((r.ego.x-a.x)*dx+(r.ego.y-a.y)*dy)/(length*length),0.,1.);
    Pose q=interpolate(a,b,f);
    if (std::abs(wrap(r.ego.yaw-q.yaw))>p_.max_heading) continue;
    double d=distance(r.ego,q);
    if (d<best-1e-8) {
      best=d; best_s=a.s+f*length; best_part=a.part; projection=q;
      lateral=(-(r.ego.x-q.x)*dy+(r.ego.y-q.y)*dx)/length;
    }
  }
  if (best_part<0||best>p_.max_distance) {
    steering_previous_=false; r.state="off_route_or_heading"; return r;
  }
  if (!acquired_) { part_=best_part; progress_=best_s; acquired_=true; }
  else progress_=std::max(progress_,best_s);
  double end=progress_;
  for (const auto &point:points_) if (point.part==part_) end=point.s;
  r.valid=true; r.projection=projection;
  r.lateral_error=lateral; r.heading_error=wrap(r.ego.yaw-projection.yaw);
  r.progress=progress_; r.remaining=std::max(0.,end-progress_);
  r.lookahead=std::clamp(p_.min_lookahead+p_.lookahead_gain*speed_,p_.min_lookahead,p_.max_lookahead);
  r.target=at(std::min(end,progress_+r.lookahead),part_);
  for (double s=std::max(points_.front().s,progress_-3);s<end&&s<=progress_+80;s+=1.)
    r.horizon.push_back(at(s,part_));
  r.horizon.push_back(at(std::min(end,progress_+80),part_));
  double dx=r.target.x-r.ego.x,dy=r.target.y-r.ego.y;
  double forward=std::cos(r.ego.yaw)*dx+std::sin(r.ego.yaw)*dy;
  double left=-std::sin(r.ego.yaw)*dx+std::cos(r.ego.yaw)*dy;
  double squared=dx*dx+dy*dy;
  r.finished=r.remaining<=p_.stop_margin&&distance(r.ego,at(end,part_))<1.;
  r.targets_valid=p_.calibrated&&r.speed_valid&&(forward>.05||r.finished);
  r.state=!p_.calibrated?"sensor_path_only_vehicle_transform_unset":
          !r.speed_valid?"waiting_for_speed":r.finished?"endpoint":
          forward<=.05?"lookahead_behind_vehicle":"tracking";
  if (r.targets_valid) {
    r.curvature=r.finished?0:2*left/std::max(squared,1e-6);
    double requested=std::atan(p_.wheelbase*r.curvature);
    double steering=std::clamp(requested,-p_.max_steering,p_.max_steering);
    r.saturated=std::abs(requested-steering)>1e-9;
    if (steering_previous_) steering=std::clamp(steering,steering_-p_.max_steering_rate*dt,steering_+p_.max_steering_rate*dt);
    steering_=r.steering=steering; steering_previous_=true;
    r.target_speed=std::min(p_.max_speed,std::sqrt(2*p_.deceleration*std::max(0.,r.remaining-p_.stop_margin)));
    if (std::abs(r.curvature)>1e-6) r.target_speed=std::min(r.target_speed,std::sqrt(p_.lateral_acceleration/std::abs(r.curvature)));
    if (r.finished) r.target_speed=0;
  } else steering_previous_=false;
  return r;
}
}
