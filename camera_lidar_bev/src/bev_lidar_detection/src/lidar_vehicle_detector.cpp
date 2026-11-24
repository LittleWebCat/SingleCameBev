#include "bev_lidar_detection/centerpoint_detector.hpp"
#include "bev_lidar_detection/detector.hpp"
#include "bev_lidar_detection/sweep_accumulator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace bev_lidar_detection {
namespace {
using Clock = std::chrono::steady_clock;
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using Marker = visualization_msgs::msg::Marker;
using Markers = visualization_msgs::msg::MarkerArray;
using Detections = vision_msgs::msg::Detection3DArray;
using Status = diagnostic_msgs::msg::DiagnosticStatus;
std::int64_t stamp_ns(const std_msgs::msg::Header &h) {
  return static_cast<std::int64_t>(h.stamp.sec)*1000000000LL+h.stamp.nanosec;
}
bool host_big_endian() {
  const std::uint16_t word=1;
  return *reinterpret_cast<const std::uint8_t *>(&word)==0;
}
template<class T> T read(const std::uint8_t *bytes, bool big_endian) {
  std::array<std::uint8_t,sizeof(T)> storage{};
  std::copy_n(bytes,sizeof(T),storage.begin());
  if (big_endian!=host_big_endian()) std::reverse(storage.begin(),storage.end());
  T value; std::memcpy(&value,storage.data(),sizeof(T)); return value;
}
template<class T> void write(std::uint8_t *bytes, T value) {
  std::array<std::uint8_t,sizeof(T)> storage{};
  std::memcpy(storage.data(),&value,sizeof(T));
  if (host_big_endian()) std::reverse(storage.begin(),storage.end());
  std::copy(storage.begin(),storage.end(),bytes);
}
const Field &field(const Cloud &cloud, const std::string &name, bool xyz=false) {
  const Field *found=nullptr;
  for (const auto &f : cloud.fields) if (f.name==name) {
    if (found) throw std::invalid_argument("Duplicate cloud field "+name);
    found=&f;
  }
  if (!found || found->count!=1) throw std::invalid_argument("Missing/non-scalar cloud field "+name);
  const auto type=found->datatype;
  if ((xyz && type!=Field::FLOAT32) || (!xyz && type!=Field::UINT8 && type!=Field::UINT16 && type!=Field::FLOAT32))
    throw std::invalid_argument("Unsupported cloud field type for "+name);
  const std::uint32_t bytes=type==Field::UINT8 ? 1 : type==Field::UINT16 ? 2 : 4;
  if (static_cast<std::uint64_t>(found->offset)+bytes>cloud.point_step)
    throw std::invalid_argument("Cloud field exceeds point_step: "+name);
  return *found;
}
// Scan end = header stamp (scan start) + largest per-point t offset, the time FAST-LIO stamps
// its pose with. Returns 0 when the cloud has no usable uint32 t field.
std::int64_t scan_end_ns(const Cloud &c) {
  const Field *t=nullptr;
  for (const auto &f : c.fields) if (f.name=="t" && f.datatype==Field::UINT32 && f.count==1) t=&f;
  if (!t || static_cast<std::uint64_t>(t->offset)+4>c.point_step) return 0;
  std::uint32_t last=0;
  for (std::uint32_t row=0;row<c.height;++row) for (std::uint32_t col=0;col<c.width;++col)
    last=std::max(last,read<std::uint32_t>(c.data.data()+static_cast<std::size_t>(row)*c.row_step+
                                           static_cast<std::size_t>(col)*c.point_step+t->offset,c.is_bigendian));
  if (!last || last>250000000U) return 0;  // not a 50/100 ms revolution in nanoseconds
  return stamp_ns(c.header)+last;
}
std::array<double,4> slerp(std::array<double,4> a,std::array<double,4> b,double f) {
  double d=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
  if (d<0) {for (auto &v:b) v=-v; d=-d;}
  std::array<double,4> q{};
  if (d>0.9995) {
    double n=0;for (int i=0;i<4;++i){q[i]=a[i]+f*(b[i]-a[i]);n+=q[i]*q[i];}
    n=std::sqrt(n);for (auto &v:q) v/=n;return q;
  }
  const double th=std::acos(d),s=std::sin(th),wa=std::sin((1-f)*th)/s,wb=std::sin(f*th)/s;
  for (int i=0;i<4;++i) q[i]=wa*a[i]+wb*b[i];
  return q;
}
using Quat=std::array<double,4>;  // x, y, z, w
Quat qmul(const Quat &a,const Quat &b) {
  return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1], a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
          a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3], a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}
Quat qconj(const Quat &q) {return {-q[0],-q[1],-q[2],q[3]};}
Quat from_rotvec(const std::array<double,3> &v) {
  const double a=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  if (a<1e-12) return {v[0]/2,v[1]/2,v[2]/2,1};
  const double s=std::sin(a/2)/a;return {v[0]*s,v[1]*s,v[2]*s,std::cos(a/2)};
}
std::array<double,3> to_rotvec(Quat q) {
  if (q[3]<0) for (auto &v:q) v=-v;
  const double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
  if (n<1e-12) return {2*q[0],2*q[1],2*q[2]};
  const double a=2*std::atan2(n,q[3]);return {q[0]*a/n,q[1]*a/n,q[2]*a/n};
}
std::array<double,3> rotate(const Quat &q,const std::array<double,3> &v) {
  const auto r=qmul(qmul(q,{v[0],v[1],v[2],0}),qconj(q));return {r[0],r[1],r[2]};
}
geometry_msgs::msg::Point point(double x,double y,double z) {
  geometry_msgs::msg::Point p; p.x=x;p.y=y;p.z=z;return p;
}
// Orange cars, magenta people (as before); other classes get their own hues.
std::uint32_t class_rgb(const std::string &name) {
  if (name=="person") return 0x00ff00ff;
  if (name=="truck" || name=="bus" || name=="trailer" || name=="construction_vehicle") return 0x00ffd700;
  if (name=="motorcycle" || name=="bicycle" || name=="cyclist") return 0x0000e5ff;
  if (name=="barrier" || name=="traffic_cone") return 0x00c0c0c0;
  return 0x00ffa500;
}
bool is_vehicle(const std::string &name) {
  return name=="car" || name=="truck" || name=="bus" || name=="trailer" || name=="construction_vehicle";
}
void class_color(Marker &m,const std::string &name) {
  const auto rgb=class_rgb(name);
  m.color.r=((rgb>>16)&0xff)/255.0F;m.color.g=((rgb>>8)&0xff)/255.0F;m.color.b=(rgb&0xff)/255.0F;m.color.a=1;
}
// The 12 edges of an axis-aligned box of half sizes (hx, hy, hz) around the marker pose,
// plus a line from the center to the front face showing the heading.
void add_box_edges(Marker &m,double hx,double hy,double hz) {
  const std::array<std::array<double,3>,8> c{{{-hx,-hy,-hz},{hx,-hy,-hz},{hx,hy,-hz},{-hx,hy,-hz},
                                              {-hx,-hy,hz},{hx,-hy,hz},{hx,hy,hz},{-hx,hy,hz}}};
  for (const auto &e : std::array<std::array<int,2>,12>{{{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}}}) {
    m.points.push_back(point(c[e[0]][0],c[e[0]][1],c[e[0]][2]));
    m.points.push_back(point(c[e[1]][0],c[e[1]][1],c[e[1]][2]));
  }
  m.points.push_back(point(0,0,0));m.points.push_back(point(hx,0,0));
}
Markers empty_markers(const std_msgs::msg::Header &h) {
  Marker clear;clear.header=h;clear.action=Marker::DELETEALL;
  Markers m;m.markers.push_back(clear);return m;
}
Cloud empty_points(const std_msgs::msg::Header &h) {
  Cloud c;c.header=h;c.height=1;c.point_step=20;c.is_dense=true;c.is_bigendian=false;
  for (const auto &item : std::array<std::pair<const char *,std::uint8_t>,5>{{
      {"x",Field::FLOAT32},{"y",Field::FLOAT32},{"z",Field::FLOAT32},
      {"rgb",Field::FLOAT32},{"source_index",Field::UINT32}}}) {
    Field f;f.name=item.first;f.datatype=item.second;f.count=1;
    f.offset=static_cast<std::uint32_t>(c.fields.size())*4;c.fields.push_back(f);
  }
  return c;
}
}  // namespace

class LidarVehicleDetector final : public rclcpp::Node {
 public:
  LidarVehicleDetector() : Node("lidar_vehicle_detector") {
    const auto engine=declare_parameter("engine_file",std::string{});
    const auto input=declare_parameter("input_topic","/points");
    const auto detections=declare_parameter("detections_topic","/camera_lidar_bev/lidar_detections_3d");
    const auto markers=declare_parameter("markers_topic","/camera_lidar_bev/lidar_vehicle_markers");
    // Full 3D wireframe boxes at the predicted height, for a perspective (orbit) view.
    const auto markers_3d=declare_parameter("markers_3d_topic","/camera_lidar_bev/lidar_boxes_3d_markers");
    const auto points=declare_parameter("points_topic","/camera_lidar_bev/lidar_vehicle_points");
    frame_=declare_parameter("output_frame","os_sensor");
    intensity_field_=declare_parameter("intensity_field","reflectivity");
    intensity_scale_=declare_parameter("intensity_scale",255.0);
    model_type_=declare_parameter("model_type",std::string{"pointpillars"});
    z_offset_=declare_parameter("z_offset_m",0.0);
    score_=declare_parameter("score_threshold",0.35);
    rate_=declare_parameter("max_publish_hz",20.0);
    stale_=declare_parameter("stale_after_s",0.5);
    min_forward_=declare_parameter("min_forward_m",0.0);
    max_forward_=declare_parameter("max_forward_m",60.0);
    const auto classes=declare_parameter<std::vector<std::string>>("classes",{"car","person"});
    // Multi-sweep (CenterPoint only): merge this many recent scans using per-scan odometry poses.
    sweeps_=static_cast<int>(declare_parameter("sweeps",1));
    sweep_span_=declare_parameter("sweep_max_span_s",0.35);
    pose_predict_max_=declare_parameter("pose_predict_max_s",0.25);
    odometry_topic_=declare_parameter("odometry_topic",std::string{"/camera_lidar_bev/mapping/odometry"});
    // Rotation of the predicted current pose from the Ouster gyro (bias estimated online).
    use_gyro_=declare_parameter("use_imu_gyro",true);
    imu_topic_=declare_parameter("imu_topic",std::string{"/imu"});
    // Optional per-class confidence thresholds, one per `classes` entry; empty uses score_threshold for all.
    const auto class_scores=declare_parameter<std::vector<double>>("class_score_thresholds",std::vector<double>{});
    mask_=declare_parameter("self_mask_enabled",true);
    mask_min_={declare_parameter("self_mask_min_x_m",-2.131),declare_parameter("self_mask_min_y_m",-0.905),declare_parameter("self_mask_min_z_m",-1.8)};
    mask_max_={declare_parameter("self_mask_max_x_m",2.350),declare_parameter("self_mask_max_y_m",0.885),declare_parameter("self_mask_max_z_m",0.0)};
    if (engine.empty() || input.empty() || detections.empty() || markers.empty() || points.empty() ||
        frame_!="os_sensor" || intensity_field_.empty() || !std::isfinite(intensity_scale_) || intensity_scale_<=0 ||
        !std::isfinite(score_) || score_<=0 || score_>=1 || !std::isfinite(rate_) || rate_<=0 || rate_>20 ||
        !std::isfinite(stale_) || stale_<=0 || stale_>0.75 || (model_type_!="pointpillars" && model_type_!="centerpoint"))
      throw std::invalid_argument("Invalid trained LiDAR detector parameters");
    // PointPillars (KITTI) only sees 0..69.12 m forward; CenterPoint (nuScenes) sees +-51.2 m around.
    const double range_min=model_type_=="centerpoint" ? -51.2 : 0.0, range_max=model_type_=="centerpoint" ? 51.2 : 69.12;
    if (!std::isfinite(min_forward_) || !std::isfinite(max_forward_) || min_forward_<range_min ||
        max_forward_>range_max || min_forward_>=max_forward_ || max_forward_<=0)
      throw std::invalid_argument("min/max_forward_m outside the model's trained range");
    for (int i=0;i<3;++i) if (!std::isfinite(mask_min_[i]) || !std::isfinite(mask_max_[i]) || mask_min_[i]>mask_max_[i])
      throw std::invalid_argument("Invalid measured vehicle self mask");
    if (!class_scores.empty() && class_scores.size()!=classes.size())
      throw std::invalid_argument("class_score_thresholds needs one value per classes entry");
    for (const double t:class_scores) if (!std::isfinite(t) || t<=0 || t>=1)
      throw std::invalid_argument("class_score_thresholds values must be in (0, 1)");
    // The detector keeps everything at or above the lowest threshold; the node applies each class's own.
    const double detector_threshold=class_scores.empty() ? score_ : *std::min_element(class_scores.begin(),class_scores.end());
    if (model_type_=="centerpoint") {
      if (!std::isfinite(z_offset_) || std::abs(z_offset_)>3) throw std::invalid_argument("Invalid z_offset_m");
      if (detector_threshold<0.1)
        throw std::invalid_argument("CenterPoint score thresholds must be >= 0.1 (the model's candidate floor)");
      CenterPointConfig config;config.engine_path=engine;config.score_threshold=static_cast<float>(detector_threshold);
      config.max_output=200;config.z_offset=static_cast<float>(z_offset_);
      auto centerpoint=std::make_unique<CenterPointDetector>(config);
      centerpoint_=centerpoint.get();
      detector_=std::move(centerpoint);
    } else {
      detector_=std::make_unique<Detector>(engine,static_cast<float>(detector_threshold),200);
    }
    const auto &names=detector_->classNames();
    enabled_classes_.assign(names.size(),false);
    class_threshold_.assign(names.size(),1.0);
    for(std::size_t i=0;i<classes.size();++i) {
      const auto it=std::find(names.begin(),names.end(),classes[i]);
      if(it==names.end() || enabled_classes_[static_cast<std::size_t>(it-names.begin())])
        throw std::invalid_argument("classes must be unique names supported by "+model_type_+": "+classes[i]);
      enabled_classes_[static_cast<std::size_t>(it-names.begin())]=true;
      class_threshold_[static_cast<std::size_t>(it-names.begin())]=class_scores.empty() ? score_ : class_scores[i];
    }
    if(classes.empty())throw std::invalid_argument("At least one output class is required");
    detections_pub_=create_publisher<Detections>(detections,rclcpp::QoS(1).reliable());
    markers_pub_=create_publisher<Markers>(markers,rclcpp::QoS(1).reliable());
    markers_3d_pub_=create_publisher<Markers>(markers_3d,rclcpp::QoS(1).reliable());
    // Own vehicle outline from the measured self-mask envelope (os_sensor +x is the car's front).
    // Latched so a viewer started later still receives it.
    if (declare_parameter("ego_vehicle_marker",true)) {
      ego_pub_=create_publisher<Markers>(declare_parameter("ego_marker_topic","/camera_lidar_bev/ego_vehicle_marker"),
                                         rclcpp::QoS(1).reliable().transient_local());
      ego_pub_->publish(ego_markers());
    }
    points_pub_=create_publisher<Cloud>(points,rclcpp::QoS(1).reliable());
    diagnostics_pub_=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics",rclcpp::QoS(5));
    cloud_sub_=create_subscription<Cloud>(input,rclcpp::SensorDataQoS().keep_last(1),
      [this](Cloud::ConstSharedPtr c){latest_=std::move(c);receipt_=Clock::now();++version_;++received_;});
    tf_sub_=create_subscription<tf2_msgs::msg::TFMessage>("/tf_static",rclcpp::QoS(10).reliable().transient_local(),
      [this](tf2_msgs::msg::TFMessage::ConstSharedPtr m){observe_transform(*m);});
    if (sweeps_<1 || sweeps_>10 || !std::isfinite(sweep_span_) || sweep_span_<0 || sweep_span_>1 ||
        !std::isfinite(pose_predict_max_) || pose_predict_max_<0 || pose_predict_max_>1.0)
      throw std::invalid_argument("Invalid sweeps/sweep_max_span_s/pose_predict_max_s");
    if (sweeps_>1) {
      if (!centerpoint_) throw std::invalid_argument("sweeps > 1 requires model_type centerpoint");
      accumulator_=std::make_unique<SweepAccumulator>(static_cast<std::size_t>(sweeps_),sweep_span_);
      odom_sub_=create_subscription<nav_msgs::msg::Odometry>(odometry_topic_,rclcpp::QoS(50).reliable(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr m){on_odometry(*m);});
      if (use_gyro_) imu_sub_=create_subscription<sensor_msgs::msg::Imu>(imu_topic_,rclcpp::SensorDataQoS().keep_last(400),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr m){on_imu(*m);});
    }
    timer_=create_wall_timer(std::chrono::duration<double>(1.0/rate_),[this]{process();});
    diag_timer_=create_wall_timer(std::chrono::seconds(1),[this]{diagnostics();});
    RCLCPP_WARN(get_logger(),"%s predictions: complete dimensions/yaw; Ouster accuracy and reflectivity adaptation remain unvalidated. Flat BEV display, no vehicle TF or tracking.",detector_->modelName());
  }

 private:
  void observe_transform(const tf2_msgs::msg::TFMessage &message) {
    for (const auto &t : message.transforms) {
      if (t.header.frame_id=="os_sensor" && t.child_frame_id=="os_imu") {
        const auto &q=t.transform.rotation;sensor_from_imu_={q.x,q.y,q.z,q.w};imu_tf_valid_=true;continue;
      }
      if (t.header.frame_id!="os_sensor" || t.child_frame_id!="os_lidar") continue;
      const auto &q=t.transform.rotation;const auto &p=t.transform.translation;
      const auto close=[](double a,double b){return std::isfinite(a) && std::abs(a-b)<1e-6;};
      // The actual Ouster driver transform used throughout this project.
      if (!close(p.x,0) || !close(p.y,0) || !close(p.z,0.038195) ||
          !close(q.x,0) || !close(q.y,0) || !close(std::abs(q.z),1) || !close(q.w,0)) {
        tf_conflict_=true;return;
      }
      tf_valid_=true;
    }
  }
  bool masked(double x,double y,double z) const {
    return mask_ && x>=mask_min_[0] && x<=mask_max_[0] && y>=mask_min_[1] && y<=mask_max_[1] && z>=mask_min_[2] && z<=mask_max_[2];
  }
  bool fresh(const Cloud &c,Clock::time_point receipt) const {
    const auto age=static_cast<double>(now().nanoseconds()-stamp_ns(c.header))/1e9;
    return c.header.stamp.nanosec<1000000000U && stamp_ns(c.header)>0 && age>=-0.05 && age<=stale_ &&
      std::chrono::duration<double>(Clock::now()-receipt).count()<=stale_;
  }
  void clear(const std::string &reason) {
    if (!cleared_ || reason!=error_) {
      std_msgs::msg::Header h;h.frame_id=frame_;h.stamp=now();
      Detections d;d.header=h;detections_pub_->publish(std::move(d));
      markers_pub_->publish(empty_markers(h));markers_3d_pub_->publish(empty_markers(h));
      points_pub_->publish(empty_points(h));++clear_count_;
    }
    cleared_=true;error_=reason;last_cars_=0;last_people_=0;last_others_=0;last_point_count_=0;
  }
  void decode(const Cloud &c,std::vector<PointXYZI> &points,std::vector<std::uint32_t> &indices) {
    const auto count=static_cast<std::uint64_t>(c.width)*c.height;
    if (!count || count>detector_->maxPoints() || !c.point_step || c.point_step>256 ||
        static_cast<std::uint64_t>(c.width)*c.point_step>c.row_step ||
        static_cast<std::uint64_t>(c.row_step)*c.height!=c.data.size())
      throw std::invalid_argument("Empty, invalid or oversized LiDAR cloud layout");
    if (c.header.frame_id!="os_lidar" && c.header.frame_id!="os_sensor")
      throw std::invalid_argument("Raw LiDAR cloud must use the verified os_lidar/os_sensor frame");
    const auto fx=field(c,"x",true);const auto fy=field(c,"y",true);const auto fz=field(c,"z",true);
    const auto fi=field(c,intensity_field_);
    const std::array<const Field *,4> used{&fx,&fy,&fz,&fi};
    for (std::size_t i=0;i<used.size();++i) for (std::size_t j=i+1;j<used.size();++j) {
      const auto size=[](const Field &f){return f.datatype==Field::UINT8 ? 1U : f.datatype==Field::UINT16 ? 2U : 4U;};
      if (used[i]->offset<used[j]->offset+size(*used[j]) && used[j]->offset<used[i]->offset+size(*used[i]))
        throw std::invalid_argument("Overlapping required cloud fields");
    }
    points.reserve(count);indices.reserve(count);last_masked_=0;
    for (std::uint32_t row=0;row<c.height;++row) for (std::uint32_t col=0;col<c.width;++col) {
      const auto *b=c.data.data()+static_cast<std::size_t>(row)*c.row_step+static_cast<std::size_t>(col)*c.point_step;
      float x=read<float>(b+fx.offset,c.is_bigendian),y=read<float>(b+fy.offset,c.is_bigendian),z=read<float>(b+fz.offset,c.is_bigendian);
      const float raw=fi.datatype==Field::UINT8 ? b[fi.offset] : fi.datatype==Field::UINT16 ?
        read<std::uint16_t>(b+fi.offset,c.is_bigendian) : read<float>(b+fi.offset,c.is_bigendian);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(raw) || raw<0 ||
          raw>intensity_scale_ || x*x+y*y+z*z<0.25F) continue;
      if (c.header.frame_id=="os_lidar") {x=-x;y=-y;z+=0.038195F;}
      if (masked(x,y,z)) {++last_masked_;continue;}
      points.push_back({x,y,z,static_cast<float>(raw/intensity_scale_)});
      indices.push_back(row*c.width+col);
    }
  }
  bool publish(const std::vector<Box3D> &boxes,const std::vector<PointXYZI> &points,
               const std::vector<std::uint32_t> &indices,const Cloud &source,Clock::time_point receipt) {
    auto h=source.header;h.frame_id=frame_;
    Detections detections;detections.header=h;
    auto markers=empty_markers(h);auto markers_3d=empty_markers(h);auto cloud=empty_points(h);
    for (std::size_t i=0;i<boxes.size();++i) {
      const auto &b=boxes[i];const auto &name=class_name(b.class_id);
      vision_msgs::msg::Detection3D d;d.header=h;
      d.bbox.center.position=point(b.x,b.y,b.z);
      d.bbox.center.orientation.z=std::sin(b.yaw/2);d.bbox.center.orientation.w=std::cos(b.yaw/2);
      d.bbox.size.x=b.size_x;d.bbox.size.y=b.size_y;d.bbox.size.z=b.size_z;
      vision_msgs::msg::ObjectHypothesisWithPose result;result.hypothesis.class_id=name;result.hypothesis.score=b.score;
      result.pose.pose=d.bbox.center;d.results.push_back(std::move(result));detections.detections.push_back(std::move(d));
      Marker rectangle;rectangle.header=h;rectangle.ns="predicted_vehicle_bev_extent";rectangle.id=static_cast<int>(i*2);
      rectangle.type=Marker::LINE_STRIP;rectangle.action=Marker::ADD;rectangle.pose.orientation.w=1;
      rectangle.scale.x=0.08;rectangle.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(rectangle,name);
      const double co=std::cos(b.yaw),si=std::sin(b.yaw);
      for (const auto &sign : std::array<std::pair<int,int>,5>{{{-1,-1},{1,-1},{1,1},{-1,1},{-1,-1}}}) {
        const double dx=sign.first*b.size_x/2,dy=sign.second*b.size_y/2;
        rectangle.points.push_back(point(b.x+co*dx-si*dy,b.y+si*dx+co*dy,0.15));
      }
      markers.markers.push_back(std::move(rectangle));
      Marker label;label.header=h;label.ns="predicted_vehicle_bev_labels";label.id=static_cast<int>(i*2+1);
      label.type=Marker::TEXT_VIEW_FACING;label.action=Marker::ADD;label.pose.orientation.w=1;
      label.pose.position=point(b.x,b.y,0.17);label.scale.z=0.55;
      label.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(label,name);
      std::ostringstream text;text<<name<<' '<<std::fixed<<std::setprecision(2)<<b.score<<"\nL "<<b.size_x<<" W "<<b.size_y<<" H "<<b.size_z<<" m";
      label.text=text.str();
      if (markers_3d_pub_->get_subscription_count()) {
        Marker box=base_3d(h,"predicted_3d_boxes",static_cast<int>(i*2),Marker::LINE_LIST,name);
        box.pose.position=point(b.x,b.y,b.z);box.pose.orientation=d.bbox.center.orientation;box.scale.x=0.05;
        add_box_edges(box,b.size_x/2,b.size_y/2,b.size_z/2);
        Marker tag=base_3d(h,"predicted_3d_labels",static_cast<int>(i*2+1),Marker::TEXT_VIEW_FACING,name);
        tag.pose.position=point(b.x,b.y,b.z+b.size_z/2+0.45);tag.scale.z=0.45;tag.text=label.text;
        markers_3d.markers.push_back(std::move(box));markers_3d.markers.push_back(std::move(tag));
      }
      markers.markers.push_back(std::move(label));
    }
    if (points_pub_->get_subscription_count()) {
      struct Region {double x,y,z,hx,hy,hz,co,si,bound_x,bound_y;std::uint32_t rgb;};
      std::vector<Region> regions;regions.reserve(boxes.size());
      for (const auto &b:boxes) {
        const double co=std::cos(b.yaw),si=std::sin(b.yaw),hx=b.size_x/2,hy=b.size_y/2;
        regions.push_back({b.x,b.y,b.z,hx,hy,b.size_z/2,co,si,std::abs(co)*hx+std::abs(si)*hy,std::abs(si)*hx+std::abs(co)*hy,class_rgb(class_name(b.class_id))});
      }
      cloud.data.reserve(points.size()*cloud.point_step);
      for (std::size_t i=0;i<points.size();++i) {
        const auto &p=points[i];bool inside=false;std::uint32_t rgb=0;
        for (const auto &b:regions) {
          const double dx=p.x-b.x,dy=p.y-b.y;
          if (std::abs(p.z-b.z)>b.hz || std::abs(dx)>b.bound_x || std::abs(dy)>b.bound_y) continue;
          if (std::abs(b.co*dx+b.si*dy)<=b.hx && std::abs(-b.si*dx+b.co*dy)<=b.hy) {inside=true;rgb=b.rgb;break;}
        }
        if (!inside) continue;
        const auto start=cloud.data.size();cloud.data.resize(start+cloud.point_step);
        auto *bytes=cloud.data.data()+start;write(bytes,p.x);write(bytes+4,p.y);write(bytes+8,p.z);
        write(bytes+12,rgb);write(bytes+16,indices[i]);++cloud.width;
      }
    }
    cloud.row_step=cloud.width*cloud.point_step;last_point_count_=cloud.width;
    if (!fresh(source,receipt)) {clear("LiDAR scan became stale during output preparation");return false;}
    detections_pub_->publish(std::move(detections));markers_pub_->publish(std::move(markers));
    markers_3d_pub_->publish(std::move(markers_3d));points_pub_->publish(std::move(cloud));
    return true;
  }
  void process() {
    const auto ros_time=now().nanoseconds();
    // A looping bag restarts its source timestamps along with /clock. Only an
    // observed simulation-clock rewind may reset the monotonic source gate.
    if (get_parameter("use_sim_time").as_bool() && ros_time<last_ros_time_ &&
        last_ros_time_-ros_time>50000000LL) {
      last_ros_time_=ros_time;last_stamp_=0;
      latest_.reset();receipt_=Clock::time_point{};version_=0;processed_=0;
      last_age_=0;processing_ms_=0;last_masked_=0;++clock_rewinds_;
      poses_.clear();if (accumulator_) accumulator_->clear();
      clear("Simulation clock rewound; waiting for new LiDAR scan");
      return;
    }
    last_ros_time_=ros_time;
    if (tf_conflict_) {clear("Ouster driver TF conflicts with model coordinates");return;}
    if (!tf_valid_) {clear("Waiting for actual Ouster os_sensor <- os_lidar static TF");return;}
    if (!latest_ || !fresh(*latest_,receipt_)) {clear("LiDAR input stale or unavailable");return;}
    if (version_==processed_) return;
    const auto cloud=latest_;processed_=version_;
    if (sweeps_<=1) {run(cloud,receipt_,nullptr,0);return;}
    // FAST-LIO publishes a scan's pose ~0.1 s after the scan, so the current scan is placed with
    // a constant-velocity prediction from the latest poses; stored sweeps get their measured
    // pose later (on_odometry -> refine_pose). No pose or a stale odometry stream: one sweep.
    const auto end=scan_end_ns(*cloud);
    const bool odometry_live=!poses_.empty() &&
      std::chrono::duration<double>(Clock::now()-last_odom_receipt_).count()<1.0;
    SensorPose pose;
    if (end>0 && odometry_live && pose_at(end,pose)) {++measured_poses_;run(cloud,receipt_,&pose,end);}
    else if (end>0 && odometry_live && predict_pose(end,pose)) {++predicted_poses_;run(cloud,receipt_,&pose,end);}
    else {if (odometry_live) ++single_fallbacks_;run(cloud,receipt_,nullptr,0);}
  }
  void on_odometry(const nav_msgs::msg::Odometry &m) {
    if (m.child_frame_id!=frame_) {odometry_status_="odometry child frame is "+m.child_frame_id+", expected "+frame_;return;}
    const auto stamp=stamp_ns(m.header);
    // A new parent frame or a time rewind (looping replay) invalidates stored poses and sweeps.
    if (m.header.frame_id!=odometry_parent_ || (!poses_.empty() && stamp<=poses_.back().first)) {
      poses_.clear();accumulator_->clear();odometry_parent_=m.header.frame_id;
    }
    const auto &p=m.pose.pose;
    SensorPose pose;pose.t={p.position.x,p.position.y,p.position.z};
    pose.q={p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w};
    if (use_gyro_ && !poses_.empty()) update_gyro_bias(poses_.back().first,poses_.back().second.q,stamp,pose.q);
    poses_.emplace_back(stamp,pose);
    while (poses_.size()>64) poses_.pop_front();
    accumulator_->refine_pose(stamp,pose);
    last_odom_receipt_=Clock::now();odometry_status_="receiving "+odometry_parent_+" -> "+frame_;
    if (!poses_.empty()) last_pose_lag_ms_=static_cast<double>(now().nanoseconds()-stamp)/1e6;
  }
  // Constant-velocity extrapolation from the two newest poses, at most pose_predict_max_s ahead.
  // FAST-LIO skips scans under load (5 Hz seen in replay), so poses may be up to 1 s apart.
  bool predict_pose(std::int64_t t,SensorPose &out) const {
    if (poses_.size()<2) return false;
    const auto &[t0,a]=poses_[poses_.size()-2];const auto &[t1,b]=poses_.back();
    if (t<=t1 || t1<=t0 || t1-t0>1000000000 || static_cast<double>(t-t1)/1e9>pose_predict_max_) return false;
    const double f=static_cast<double>(t-t0)/static_cast<double>(t1-t0);
    for (int k=0;k<3;++k) out.t[k]=a.t[k]+f*(b.t[k]-a.t[k]);
    Quat delta;
    if (gyro_rotation(t1,t,true,delta)) {out.q=qmul(b.q,delta);++gyro_predictions_;}
    else out.q=slerp(a.q,b.q,f);
    return true;
  }
  void on_imu(const sensor_msgs::msg::Imu &m) {
    if (!imu_tf_valid_ || m.header.frame_id!="os_imu") return;
    const auto stamp=stamp_ns(m.header);
    if (!gyro_.empty() && stamp<=gyro_.front().first) gyro_.clear();  // replay rewind
    if (!gyro_.empty() && stamp<=gyro_.back().first) return;
    const auto &w=m.angular_velocity;
    gyro_.emplace_back(stamp,rotate(sensor_from_imu_,{w.x,w.y,w.z}));
    while (gyro_.size()>300) gyro_.pop_front();  // ~3 s at 100 Hz
  }
  // Body-frame (os_sensor) rotation from t0 to t1 by integrating the gyro, piecewise constant
  // between samples. Fails if the samples do not cover the interval (gaps > 50 ms, or more
  // than 30 ms missing at either end).
  bool gyro_rotation(std::int64_t t0,std::int64_t t1,bool remove_bias,Quat &out) const {
    if (gyro_.size()<2 || t1<=t0 || gyro_.front().first>t0+30000000 || gyro_.back().first<t1-30000000) return false;
    out={0,0,0,1};
    std::int64_t t=t0;
    for (std::size_t i=0;i<gyro_.size() && t<t1;++i) {
      const auto next=i+1<gyro_.size() ? gyro_[i+1].first : t1;
      if (next<=t) continue;
      const auto end=std::min(next,t1);
      if (i+1<gyro_.size() && next-gyro_[i].first>50000000) return false;
      const double dt=static_cast<double>(end-t)/1e9;
      std::array<double,3> w=gyro_[i].second;
      if (i+1<gyro_.size()) for (int k=0;k<3;++k) w[k]=0.5*(w[k]+gyro_[i+1].second[k]);
      if (remove_bias) for (int k=0;k<3;++k) w[k]-=gyro_bias_[k];
      out=qmul(out,from_rotvec({w[0]*dt,w[1]*dt,w[2]*dt}));
      t=end;
    }
    return t>=t1;
  }
  // Gyro bias: the gyro's rotation between two measured FAST-LIO poses minus the measured
  // rotation, per second, slowly low-pass filtered (pose rotation noise) and clamped to 3 deg/s.
  void update_gyro_bias(std::int64_t t0,const Quat &q0,std::int64_t t1,const Quat &q1) {
    Quat raw;
    if (t1-t0>1000000000 || !gyro_rotation(t0,t1,false,raw)) return;
    const auto err=to_rotvec(qmul(qconj(qmul(qconj(q0),q1)),raw));
    const double dt=static_cast<double>(t1-t0)/1e9;
    for (int k=0;k<3;++k) gyro_bias_[k]=std::clamp(gyro_bias_[k]+0.02*(err[k]/dt-gyro_bias_[k]),-0.052,0.052);
  }
  // Exact (within 1 ms) or interpolated across a gap of at most 150 ms; never extrapolated.
  bool pose_at(std::int64_t t,SensorPose &out) const {
    for (std::size_t i=0;i<poses_.size();++i) {
      if (std::llabs(poses_[i].first-t)<=1000000) {out=poses_[i].second;return true;}
      if (i>0 && poses_[i-1].first<t && t<poses_[i].first) {
        const auto &[t0,a]=poses_[i-1];const auto &[t1,b]=poses_[i];
        if (t1-t0>150000000) return false;
        const double f=static_cast<double>(t-t0)/static_cast<double>(t1-t0);
        for (int k=0;k<3;++k) out.t[k]=a.t[k]+f*(b.t[k]-a.t[k]);
        out.q=slerp(a.q,b.q,f);return true;
      }
    }
    return false;
  }
  // pose (with end_ns, the scan-end time it refers to) enables multi-sweep; null runs one sweep.
  void run(const Cloud::ConstSharedPtr &cloud,Clock::time_point receipt,const SensorPose *pose,std::int64_t end_ns) {
    const auto start=Clock::now();
    const auto stamp=stamp_ns(cloud->header);
    if (stamp<=last_stamp_) {clear("Repeated/nonmonotonic LiDAR source timestamp");return;}
    last_stamp_=stamp;
    try {
      std::vector<PointXYZI> points;std::vector<std::uint32_t> indices;decode(*cloud,points,indices);
      std::vector<Box3D> predicted;
      if (accumulator_ && pose) {
        accumulator_->add(points,end_ns,*pose);
        accumulator_->build(merged_,ages_);
        predicted=centerpoint_->infer(merged_,ages_);
        last_sweeps_used_=accumulator_->sweeps();++multi_sweep_count_;
      } else {
        if (accumulator_) accumulator_->clear();  // a gap without pose breaks the sweep chain
        predicted=detector_->infer(points);last_sweeps_used_=1;
      }
      std::vector<Box3D> objects;
      for (const auto &b:predicted) {
        if (b.class_id<0 || static_cast<std::size_t>(b.class_id)>=enabled_classes_.size() || !enabled_classes_[b.class_id] ||
            b.score<class_threshold_[b.class_id] || b.score>1 || b.x<min_forward_ || b.x>max_forward_ ||
            b.size_x>20 || b.size_y>6 || b.size_z>5 || masked(b.x,b.y,b.z)) continue;
        objects.push_back(b);
      }
      if (!fresh(*cloud,receipt)) {clear("LiDAR scan became stale during inference");return;}
      if (!publish(objects,points,indices,*cloud,receipt)) return;
      last_cars_=static_cast<std::size_t>(std::count_if(objects.begin(),objects.end(),[this](const auto &b){return is_vehicle(class_name(b.class_id));}));
      last_people_=static_cast<std::size_t>(std::count_if(objects.begin(),objects.end(),[this](const auto &b){return class_name(b.class_id)=="person";}));
      last_others_=objects.size()-last_cars_-last_people_;++processed_count_;
      cleared_=false;error_.clear();last_age_=static_cast<double>(now().nanoseconds()-stamp)/1e9;
      processing_ms_=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    } catch (const std::exception &e) {clear(e.what());}
  }
  static Marker base_3d(const std_msgs::msg::Header &h,const char *ns,int id,int type,const std::string &name) {
    Marker m;m.header=h;m.ns=ns;m.id=id;m.type=type;m.action=Marker::ADD;m.pose.orientation.w=1;
    m.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(m,name);return m;
  }
  Markers ego_markers() const {
    Markers out;
    std_msgs::msg::Header h;h.frame_id=frame_;h.stamp=now();
    const double x0=mask_min_[0],x1=mask_max_[0],y0=mask_min_[1],y1=mask_max_[1];
    const auto base=[&](int id,int type){
      Marker m;m.header=h;m.ns="ego_vehicle";m.id=id;m.type=type;m.action=Marker::ADD;
      m.pose.orientation.w=1;m.frame_locked=true;return m;
    };
    auto body=base(0,Marker::CUBE);  // translucent footprint, flat like the detection boxes
    body.pose.position=point((x0+x1)/2,(y0+y1)/2,0.1);
    body.scale.x=x1-x0;body.scale.y=y1-y0;body.scale.z=0.02;
    body.color.r=0.35F;body.color.g=0.75F;body.color.b=1.0F;body.color.a=0.35F;
    auto outline=base(1,Marker::LINE_STRIP);
    outline.scale.x=0.08;outline.color.r=0.35F;outline.color.g=0.75F;outline.color.b=1.0F;outline.color.a=1.0F;
    for (const auto &c : std::array<std::array<double,2>,5>{{{x0,y0},{x1,y0},{x1,y1},{x0,y1},{x0,y0}}})
      outline.points.push_back(point(c[0],c[1],0.15));
    auto heading=base(2,Marker::ARROW);  // from the LiDAR toward the front bumper
    heading.points={point(0,(y0+y1)/2,0.16),point(x1+1.0,(y0+y1)/2,0.16)};
    heading.scale.x=0.12;heading.scale.y=0.35;heading.scale.z=0.35;heading.color=outline.color;
    auto label=base(3,Marker::TEXT_VIEW_FACING);
    label.pose.position=point((x0+x1)/2,(y0+y1)/2,0.2);label.scale.z=0.5;label.color=outline.color;
    std::ostringstream text;text<<std::fixed<<std::setprecision(2)<<"own vehicle\n"<<x1-x0<<" x "<<y1-y0<<" m";
    label.text=text.str();
    auto box=base(4,Marker::LINE_LIST);  // 3D outline over the measured envelope, for the orbit view
    box.ns="ego_vehicle_3d";box.pose.position=point((x0+x1)/2,(y0+y1)/2,(mask_min_[2]+mask_max_[2])/2);
    box.scale.x=0.05;box.color=outline.color;
    add_box_edges(box,(x1-x0)/2,(y1-y0)/2,(mask_max_[2]-mask_min_[2])/2);
    out.markers={body,outline,heading,label,box};
    return out;
  }
  void diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message;message.header.stamp=now();
    Status s;s.name=get_name();  // "lidar_vehicle_detector" unless renamed for a side-by-side runs.hardware_id="ouster+Thor";s.level=Status::WARN;
    s.message=error_.empty() ? std::string(detector_->modelName())+" predictions active; Ouster accuracy remains provisional" : error_;
    const auto add=[&s](const std::string &key,const std::string &value){diagnostic_msgs::msg::KeyValue v;v.key=key;v.value=value;s.values.push_back(v);};
    const auto &stats=detector_->lastStats();
    add("model",detector_->modelName());add("trained_3d_detector","true");add("full_dimensions_predicted","true");
    add("heading_predicted","true");add("physical_accuracy_validated","false");add("tracking_enabled","false");
    add("frame",frame_);add("driver_tf_valid",tf_valid_ && !tf_conflict_ ? "true" : "false");
    add("publishing_fresh_output",cleared_ ? "false" : "true");
    std::string enabled;
    std::ostringstream thresholds;thresholds<<std::fixed<<std::setprecision(2);
    for (std::size_t i=0;i<enabled_classes_.size();++i) if (enabled_classes_[i]) {
      enabled+=(enabled.empty() ? "" : ",")+detector_->classNames()[i];
      thresholds<<(thresholds.tellp()>0 ? "," : "")<<detector_->classNames()[i]<<'='<<class_threshold_[i];
    }
    add("enabled_classes",enabled);add("class_score_thresholds",thresholds.str());
    add("intensity_field",intensity_field_);add("intensity_scale",std::to_string(intensity_scale_));
    add("received_scans",std::to_string(received_));add("processed_scans",std::to_string(processed_count_));
    add("simulation_clock_rewinds",std::to_string(clock_rewinds_));
    add("cleared_outputs",std::to_string(clear_count_));add("vehicle_boxes",std::to_string(last_cars_));
    add("person_boxes",std::to_string(last_people_));add("other_boxes",std::to_string(last_others_));
    add("object_boxes",std::to_string(last_cars_+last_people_+last_others_));
    add("vehicle_points",std::to_string(last_point_count_));add("self_masked_points",std::to_string(last_masked_));
    add("model_input_points",std::to_string(stats.accepted_points));add("detector_ms",std::to_string(stats.processing_ms));
    add("input_pillars",std::to_string(stats.input_pillars));add("accepted_pillars",std::to_string(stats.accepted_pillars));
    add("truncated_pillars",std::to_string(stats.truncated_pillars));add("truncated_points",std::to_string(stats.truncated_points));
    add("sweeps_configured",std::to_string(sweeps_));add("sweeps_used",std::to_string(last_sweeps_used_));
    add("odometry_topic",sweeps_>1 ? odometry_topic_ : "not used");add("odometry_status",odometry_status_);
    add("odometry_lag_ms",std::to_string(last_pose_lag_ms_));add("multi_sweep_scans",std::to_string(multi_sweep_count_));
    add("current_pose_measured",std::to_string(measured_poses_));add("current_pose_predicted",std::to_string(predicted_poses_));
    add("gyro_rotation_predictions",std::to_string(gyro_predictions_));
    {std::ostringstream b;b<<std::fixed<<std::setprecision(3)<<gyro_bias_[0]*57.2958<<','<<gyro_bias_[1]*57.2958<<','<<gyro_bias_[2]*57.2958;
     add("gyro_bias_deg_s",use_gyro_ ? b.str() : "not used");}
    add("single_sweep_fallbacks",std::to_string(single_fallbacks_));
    add("preprocess_ms",std::to_string(stats.preprocess_ms));add("inference_ms",std::to_string(stats.inference_ms));
    add("postprocess_ms",std::to_string(stats.postprocess_ms));
    add("processing_ms",std::to_string(processing_ms_));add("source_age_s",std::to_string(last_age_));
    add("publication_cap_hz",std::to_string(rate_));message.status.push_back(std::move(s));diagnostics_pub_->publish(std::move(message));
  }
  const std::string &class_name(int id) const {return detector_->classNames().at(static_cast<std::size_t>(id));}
  std::unique_ptr<ObjectDetector> detector_;
  CenterPointDetector *centerpoint_=nullptr;
  // Multi-sweep state.
  int sweeps_{1};double sweep_span_{},pose_predict_max_{},last_pose_lag_ms_{};
  std::string odometry_topic_,odometry_parent_,odometry_status_{"not used"};
  std::unique_ptr<SweepAccumulator> accumulator_;
  std::deque<std::pair<std::int64_t,SensorPose>> poses_;
  Clock::time_point last_odom_receipt_{};
  std::vector<PointXYZI> merged_;std::vector<float> ages_;
  std::size_t last_sweeps_used_{1};std::uint64_t multi_sweep_count_{},single_fallbacks_{},measured_poses_{},predicted_poses_{};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  // Gyro-based rotation prediction.
  bool use_gyro_{true},imu_tf_valid_{false};std::string imu_topic_;
  Quat sensor_from_imu_{0,0,0,1};std::array<double,3> gyro_bias_{0,0,0};
  std::deque<std::pair<std::int64_t,std::array<double,3>>> gyro_;
  mutable std::uint64_t gyro_predictions_{};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  std::string frame_,intensity_field_,model_type_,error_{"Waiting for LiDAR"};
  double intensity_scale_{},score_{},rate_{},stale_{},min_forward_{},max_forward_{},z_offset_{},last_age_{},processing_ms_{};
  std::array<double,3> mask_min_{},mask_max_{};
  std::vector<bool> enabled_classes_;
  std::vector<double> class_threshold_;
  bool mask_{},tf_valid_{false},tf_conflict_{false},cleared_{false};
  Cloud::ConstSharedPtr latest_;Clock::time_point receipt_{};
  std::uint64_t version_{},processed_{},received_{},processed_count_{},clear_count_{},clock_rewinds_{};
  std::int64_t last_stamp_{0},last_ros_time_{0};std::size_t last_masked_{},last_cars_{},last_people_{},last_others_{},last_point_count_{};
  rclcpp::Publisher<Detections>::SharedPtr detections_pub_;
  rclcpp::Publisher<Markers>::SharedPtr markers_pub_,markers_3d_pub_,ego_pub_;
  rclcpp::Publisher<Cloud>::SharedPtr points_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<Cloud>::SharedPtr cloud_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::TimerBase::SharedPtr timer_,diag_timer_;
};
}  // namespace bev_lidar_detection
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try {rclcpp::spin(std::make_shared<bev_lidar_detection::LidarVehicleDetector>());}
  catch (const std::exception &e) {fprintf(stderr,"lidar_vehicle_detector: %s\n",e.what());rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
