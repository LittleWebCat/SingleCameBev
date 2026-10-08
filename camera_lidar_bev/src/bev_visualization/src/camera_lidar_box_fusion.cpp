#include "bev_visualization/camera_box_association.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace bev_visualization {
namespace {
using Clock=std::chrono::steady_clock;
using Lidar=vision_msgs::msg::Detection3DArray;
using CameraBoxes=vision_msgs::msg::Detection2DArray;
using Info=sensor_msgs::msg::CameraInfo;
using Cloud=sensor_msgs::msg::PointCloud2;
using Field=sensor_msgs::msg::PointField;
using Marker=visualization_msgs::msg::Marker;
using Markers=visualization_msgs::msg::MarkerArray;
using Status=diagnostic_msgs::msg::DiagnosticStatus;
template<class M> struct Received {typename M::ConstSharedPtr message;Clock::time_point receipt;};
template<class M> void append(std::deque<Received<M>> &ring,typename M::ConstSharedPtr m,std::size_t capacity) {
  ring.push_back({std::move(m),Clock::now()});while(ring.size()>capacity)ring.pop_front();
}
std::int64_t stamp_ns(const std_msgs::msg::Header &h) {
  return static_cast<std::int64_t>(h.stamp.sec)*1000000000LL+h.stamp.nanosec;
}
bool same_header(const std_msgs::msg::Header &a,const std_msgs::msg::Header &b) {
  return a.frame_id==b.frame_id && a.stamp.sec==b.stamp.sec && a.stamp.nanosec==b.stamp.nanosec;
}
template<std::size_t N> std::array<double,N> yaml_array(const YAML::Node &node,const std::string &name) {
  if(!node.IsSequence() || node.size()!=N)throw std::invalid_argument("Invalid "+name+" array length");
  std::array<double,N> values{};
  for(std::size_t i=0;i<N;++i){values[i]=node[i].as<double>();if(!std::isfinite(values[i]))throw std::invalid_argument("Nonfinite "+name);}
  return values;
}
Transform yaml_transform(const YAML::Node &node) {
  if(!node.IsSequence() || node.size()!=4)throw std::invalid_argument("Calibration transform must be 4x4");
  Transform t{};
  for(std::size_t i=0;i<4;++i){const auto row=yaml_array<4>(node[i],"transform row");std::copy(row.begin(),row.end(),t.begin()+i*4);}
  validate_transform(t);return t;
}
template<class A,class B> bool close_arrays(const A &a,const B &b) {
  if(a.size()!=b.size())return false;
  for(std::size_t i=0;i<a.size();++i)if(!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i]-b[i])>1e-8)return false;
  return true;
}
geometry_msgs::msg::Point point(double x,double y,double z) {
  geometry_msgs::msg::Point p;p.x=x;p.y=y;p.z=z;return p;
}
Markers empty_markers(const std_msgs::msg::Header &h) {
  Marker clear;clear.header=h;clear.action=Marker::DELETEALL;
  Markers out;out.markers.push_back(clear);return out;
}
void class_color(Marker &m,const std::string &name){m.color.r=1;m.color.g=name=="person" ? 0 : 165.0F/255;m.color.b=name=="person" ? 1 : 0;m.color.a=1;}
bool host_big_endian(){const std::uint16_t v=1;return *reinterpret_cast<const std::uint8_t *>(&v)==0;}
template<class T> T read_scalar(const std::uint8_t *bytes,bool big_endian) {
  std::array<std::uint8_t,sizeof(T)> storage{};std::copy_n(bytes,sizeof(T),storage.begin());
  if(big_endian!=host_big_endian())std::reverse(storage.begin(),storage.end());
  T value;std::memcpy(&value,storage.data(),sizeof(T));return value;
}
template<class T> void write_scalar(std::uint8_t *bytes,T value) {
  std::array<std::uint8_t,sizeof(T)> storage{};std::memcpy(storage.data(),&value,sizeof(T));
    if(host_big_endian())std::reverse(storage.begin(),storage.end());
    std::copy(storage.begin(),storage.end(),bytes);
}
Cloud empty_points(const std_msgs::msg::Header &h) {
  Cloud c;c.header=h;c.height=1;c.point_step=20;c.is_dense=true;c.is_bigendian=false;
  for(const auto &entry:std::array<std::pair<const char *,std::uint8_t>,5>{{{"x",Field::FLOAT32},{"y",Field::FLOAT32},{"z",Field::FLOAT32},{"rgb",Field::FLOAT32},{"source_index",Field::UINT32}}}) {
    Field f;f.name=entry.first;f.datatype=entry.second;f.count=1;f.offset=static_cast<std::uint32_t>(c.fields.size())*4;c.fields.push_back(f);
  }
  return c;
}
bool known_camera_class(const std::string &name) {
  return name=="person" || name=="bicycle" || name=="car" || name=="motorcycle" || name=="bus" || name=="truck";
}
}  // namespace

class CameraLidarBoxFusion final : public rclcpp::Node {
 public:
  CameraLidarBoxFusion():Node("camera_lidar_box_fusion") {
    camera_frame_=declare_parameter("camera_frame","camera_front_optical_frame");
    const auto intrinsics=declare_parameter("intrinsics_file",std::string{});
    const auto extrinsics=declare_parameter("extrinsics_file",std::string{});
    const auto allow_provisional=declare_parameter("allow_provisional_calibration",false);
    pair_delta_=declare_parameter("max_pair_delta_s",0.15);
    stale_=declare_parameter("stale_after_s",0.5);
    rate_=declare_parameter("max_publish_hz",20.0);
    require_camera_match_=declare_parameter("require_camera_match",false);
    config_.min_iou=declare_parameter("min_iou",0.20);
    config_.ambiguity_margin=declare_parameter("ambiguity_margin",0.05);
    config_.min_confidence=declare_parameter("min_camera_confidence",0.35);
    if(camera_frame_.empty() || intrinsics.empty() || extrinsics.empty() ||
       !std::isfinite(pair_delta_) || pair_delta_<=0 || pair_delta_>0.15 ||
       !std::isfinite(stale_) || stale_<=0 || stale_>0.75 || !std::isfinite(rate_) || rate_<=0 || rate_>20)
      throw std::invalid_argument("Invalid camera/LiDAR association parameters");
    validate_association_config(config_);load_calibration(intrinsics,extrinsics,allow_provisional);
    const auto lidar_topic=declare_parameter("lidar_detections_topic","/camera_lidar_bev/lidar_detections_3d");
    const auto camera_topic=declare_parameter("camera_detections_topic","/camera_lidar_bev/detections_2d");
    const auto info_topic=declare_parameter("camera_info_topic","/camera/front/camera_info");
    const auto markers_topic=declare_parameter("markers_topic","/camera_lidar_bev/camera_lidar_markers");
    const auto lidar_points_topic=declare_parameter("lidar_points_topic","/camera_lidar_bev/lidar_vehicle_points");
    const auto points_topic=declare_parameter("points_topic","/camera_lidar_bev/camera_lidar_points");
    const auto output_detections_topic=declare_parameter("output_detections_topic","/camera_lidar_bev/camera_lidar_detections_3d");
    if(lidar_topic.empty() || camera_topic.empty() || info_topic.empty() || markers_topic.empty() ||
       lidar_points_topic.empty() || points_topic.empty() || output_detections_topic.empty())
      throw std::invalid_argument("Association topics must not be empty");
    markers_pub_=create_publisher<Markers>(markers_topic,rclcpp::QoS(1).reliable());
    points_pub_=create_publisher<Cloud>(points_topic,rclcpp::QoS(1).reliable());
    detections_pub_=create_publisher<Lidar>(output_detections_topic,rclcpp::QoS(1).reliable());
    diagnostics_pub_=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics",rclcpp::QoS(5));
    lidar_sub_=create_subscription<Lidar>(lidar_topic,rclcpp::QoS(1).reliable(),
      [this](Lidar::ConstSharedPtr m){lidar_={std::move(m),Clock::now()};++lidar_version_;});
    camera_sub_=create_subscription<CameraBoxes>(camera_topic,rclcpp::QoS(1).reliable(),
      [this](CameraBoxes::ConstSharedPtr m){append<CameraBoxes>(camera_ring_,std::move(m),32);++camera_version_;});
    info_sub_=create_subscription<Info>(info_topic,rclcpp::SensorDataQoS().keep_last(64),
      [this](Info::ConstSharedPtr m){append<Info>(infos_,std::move(m),64);++info_version_;});
    points_sub_=create_subscription<Cloud>(lidar_points_topic,rclcpp::QoS(1).reliable(),
      [this](Cloud::ConstSharedPtr m){append<Cloud>(point_ring_,std::move(m),8);++points_version_;});
    timer_=create_wall_timer(std::chrono::duration<double>(1.0/rate_),[this]{process();});
    diag_timer_=create_wall_timer(std::chrono::seconds(1),[this]{diagnostics();});
    RCLCPP_WARN(get_logger(),"Camera labels associated with unchanged PointPillars geometry. Camera calibration/timing remain provisional; image overlap is an association, not independent visibility confirmation.");
  }

 private:
  void load_calibration(const std::string &intrinsics,const std::string &extrinsics,bool allowed) {
    const auto native=YAML::LoadFile(intrinsics);
    if(native["distortion_model"].as<std::string>()!="plumb_bob")throw std::invalid_argument("Only native plumb_bob camera calibration is supported");
    camera_.width=native["image_width"].as<std::uint32_t>();camera_.height=native["image_height"].as<std::uint32_t>();
    camera_.k=yaml_array<9>(native["camera_matrix"]["data"],"K");
    camera_.d=yaml_array<5>(native["distortion_coefficients"]["data"],"D");
    expected_r_=yaml_array<9>(native["rectification_matrix"]["data"],"R");
    expected_p_=yaml_array<12>(native["projection_matrix"]["data"],"P");validate_camera(camera_);
    const std::array<double,9> identity{1,0,0,0,1,0,0,0,1};
    const std::array<double,12> p{camera_.k[0],0,camera_.k[2],0,0,camera_.k[4],camera_.k[5],0,0,0,1,0};
    if(!close_arrays(expected_r_,identity) || !close_arrays(expected_p_,p))throw std::invalid_argument("Raw image requires identity R and native [K|0] P");
    const auto candidate=YAML::LoadFile(extrinsics);
    if(!allowed || candidate["units"].as<std::string>()!="m" || candidate["direction"].as<std::string>()!="lidar_to_camera" ||
       candidate["selected_hypothesis"].as<std::string>()!="B" || candidate["calibration_status"].as<std::string>()!="external_candidate_unvalidated")
      throw std::invalid_argument("Explicit provisional selected-B calibration is required");
    const auto t=candidate["transforms"]["T_camera_from_os_sensor"];
    if(t["source_frame"].as<std::string>()!="os_sensor" || t["target_frame"].as<std::string>()!=camera_frame_)
      throw std::invalid_argument("Camera calibration transform frames disagree with model output");
    camera_from_sensor_=yaml_transform(t["matrix"]);
    const auto driver=candidate["observed_driver_transform"];
    if(driver["source_frame"].as<std::string>()!="os_lidar" || driver["target_frame"].as<std::string>()!="os_sensor")
      throw std::invalid_argument("Calibration driver frames are inconsistent");
    if(!close_arrays(compose(camera_from_sensor_,yaml_transform(driver["matrix"])),
                     yaml_transform(candidate["transforms"]["T_camera_from_os_lidar"]["matrix"])))
      throw std::invalid_argument("Calibration transform composition is inconsistent");
  }
  template<class M> bool fresh(const Received<M> &m) const {
    if(!m.message)return false;
    const auto &h=m.message->header;const auto stamp=stamp_ns(h);
    const double age=static_cast<double>(now().nanoseconds()-stamp)/1e9;
    return stamp>0 && h.stamp.nanosec<1000000000U && age>=-0.05 && age<=stale_ &&
      std::chrono::duration<double>(Clock::now()-m.receipt).count()<=stale_;
  }
  bool valid_info(const Info &i) const {
    return i.header.frame_id==camera_frame_ && i.width==camera_.width && i.height==camera_.height &&
      i.distortion_model=="plumb_bob" && close_arrays(i.k,camera_.k) && close_arrays(i.d,camera_.d) &&
      close_arrays(i.r,expected_r_) && close_arrays(i.p,expected_p_) && i.binning_x<=1 && i.binning_y<=1 &&
      !i.roi.x_offset && !i.roi.y_offset && !i.roi.do_rectify &&
      (!i.roi.width || i.roi.width==camera_.width) && (!i.roi.height || i.roi.height==camera_.height);
  }
  std::vector<OrientedBox> decode_lidar(const Lidar &m) const {
    if(m.header.frame_id!="os_sensor" || m.detections.size()>200)throw std::invalid_argument("LiDAR detection frame/count mismatch");
    std::vector<OrientedBox> out;out.reserve(m.detections.size());
    for(const auto &d:m.detections) {
      if(!same_header(d.header,m.header) || d.results.size()!=1 ||
         (d.results[0].hypothesis.class_id!="car" && d.results[0].hypothesis.class_id!="person") ||
         !std::isfinite(d.results[0].hypothesis.score) || d.results[0].hypothesis.score<0 || d.results[0].hypothesis.score>1)
        throw std::invalid_argument("LiDAR child header/class/score mismatch");
      const auto &b=d.bbox;const auto &p=b.center.position;const auto &q=b.center.orientation;
      const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
      if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
         !std::isfinite(b.size.x) || !std::isfinite(b.size.y) || !std::isfinite(b.size.z) ||
         b.size.x<=0 || b.size.y<=0 || b.size.z<=0 || b.size.x>15 || b.size.y>6 || b.size.z>5 ||
         !std::isfinite(norm) || std::abs(norm-1)>1e-6 || std::abs(q.x)>1e-6 || std::abs(q.y)>1e-6)
        throw std::invalid_argument("Invalid predicted vehicle dimensions/yaw quaternion");
      out.push_back({{p.x,p.y,p.z},{b.size.x,b.size.y,b.size.z},{q.x,q.y,q.z,q.w},d.results[0].hypothesis.class_id});
    }
    return out;
  }
  std::vector<CameraDetection> decode_camera(const CameraBoxes &m) const {
    if(m.header.frame_id!=camera_frame_ || m.detections.size()>100)throw std::invalid_argument("Camera frame/count mismatch");
    std::vector<CameraDetection> out;out.reserve(m.detections.size());
    for(const auto &d:m.detections) {
      if(!same_header(d.header,m.header))throw std::invalid_argument("Camera child source header mismatch");
      const auto &b=d.bbox;
      if(!std::isfinite(b.center.position.x) || !std::isfinite(b.center.position.y) || !std::isfinite(b.center.theta) ||
         std::abs(b.center.theta)>1e-8 || !std::isfinite(b.size_x) || !std::isfinite(b.size_y) || b.size_x<=0 || b.size_y<=0)
        throw std::invalid_argument("Invalid native camera ROI");
      const vision_msgs::msg::ObjectHypothesis *best=nullptr;
      for(const auto &r:d.results) {
        const auto &h=r.hypothesis;
        if(!std::isfinite(h.score) || h.score<0 || h.score>1)throw std::invalid_argument("Invalid camera confidence");
        if(known_camera_class(h.class_id) && (!best || h.score>best->score))best=&h;
      }
      const double xmin=b.center.position.x-b.size_x/2,xmax=b.center.position.x+b.size_x/2;
      const double ymin=b.center.position.y-b.size_y/2,ymax=b.center.position.y+b.size_y/2;
      if(xmin<-1e-6 || ymin<-1e-6 || xmax>camera_.width+1e-6 || ymax>camera_.height+1e-6)
        throw std::invalid_argument("Camera ROI lies outside calibrated native image");
      if(best)out.push_back({best->class_id,best->score,std::max(0.0,xmin),std::max(0.0,ymin),
                            std::min(static_cast<double>(camera_.width),xmax),std::min(static_cast<double>(camera_.height),ymax)});
    }
    return out;
  }
  void update_camera_validity() {
    if(checked_camera_==camera_version_)return;
    checked_camera_=camera_version_;camera_latest_valid_=false;
    if(camera_ring_.empty() || !fresh(camera_ring_.back()))return;
    const auto &m=*camera_ring_.back().message;const auto stamp=stamp_ns(m.header);
    if(m.header.frame_id!=camera_frame_ || stamp<=last_camera_stamp_)return;
    last_camera_stamp_=stamp;
    try {decode_camera(m);}catch(const std::exception &){return;}
    camera_latest_valid_=true;
    if(m.detections.empty())last_empty_camera_stamp_=stamp;
  }
  void clear(const std::string &reason) {
    if(!cleared_ || reason!=error_) {
      std_msgs::msg::Header h;h.frame_id="os_sensor";h.stamp=now();markers_pub_->publish(empty_markers(h));
      points_pub_->publish(empty_points(h));Lidar detections;detections.header=h;detections_pub_->publish(std::move(detections));++clear_count_;
    }
    cleared_=true;error_=reason;last_boxes_=0;last_matches_=0;last_displayed_=0;last_people_=0;
  }
  Markers render(const Lidar &lidar,const std::vector<OrientedBox> &boxes,
                 const std::vector<CameraDetection> &camera_boxes,const std::vector<BoxCameraMatch> &matches) const {
    auto out=empty_markers(lidar.header);
    for(std::size_t i=0;i<boxes.size();++i) {
      if(require_camera_match_ && matches[i].camera_index<0)continue;
      const auto &b=boxes[i];const auto &q=b.orientation_xyzw;const double yaw=2*std::atan2(q[2],q[3]);
      const double co=std::cos(yaw),si=std::sin(yaw);
      Marker rectangle;rectangle.header=lidar.header;rectangle.ns="camera_lidar_bev_extent";rectangle.id=static_cast<int>(i*2);
      rectangle.type=Marker::LINE_STRIP;rectangle.action=Marker::ADD;rectangle.pose.orientation.w=1;
      rectangle.scale.x=0.08;rectangle.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(rectangle,b.class_name);
      for(const auto &sign:std::array<std::pair<int,int>,5>{{{-1,-1},{1,-1},{1,1},{-1,1},{-1,-1}}}) {
        const double dx=sign.first*b.size.x/2,dy=sign.second*b.size.y/2;
        rectangle.points.push_back(point(b.center.x+co*dx-si*dy,b.center.y+si*dx+co*dy,0.15));
      }
      out.markers.push_back(std::move(rectangle));
      Marker label;label.header=lidar.header;label.ns="camera_lidar_labels";label.id=static_cast<int>(i*2+1);
      label.type=Marker::TEXT_VIEW_FACING;label.action=Marker::ADD;label.pose.orientation.w=1;
      label.pose.position=point(b.center.x,b.center.y,0.17);label.scale.z=0.55;
      label.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(label,b.class_name);
      std::ostringstream text;text<<b.class_name<<" L:"<<std::fixed<<std::setprecision(2)<<lidar.detections[i].results[0].hypothesis.score<<" C:";
      if(matches[i].camera_index>=0) {
        const auto &c=camera_boxes.at(static_cast<std::size_t>(matches[i].camera_index));text<<c.class_name<<' '<<c.confidence;
      } else text<<"--";
      text<<"\nL "<<b.size.x<<" W "<<b.size.y<<" H "<<b.size.z<<" m";
      label.text=text.str();out.markers.push_back(std::move(label));
    }
    return out;
  }
  Cloud filtered_points(const std::vector<OrientedBox> &boxes,const std::vector<BoxCameraMatch> &matches,
                        const std_msgs::msg::Header &header) const {
    auto out=empty_points(header);
    if(!points_pub_->get_subscription_count())return out;
    struct Region {double x,y,z,hx,hy,hz,co,si,bx,by;std::uint32_t rgb;};
    std::vector<Region> regions;
    for(std::size_t i=0;i<boxes.size();++i) {
      if(require_camera_match_ && matches[i].camera_index<0)continue;
      const auto &b=boxes[i];const double yaw=2*std::atan2(b.orientation_xyzw[2],b.orientation_xyzw[3]);
      const double co=std::cos(yaw),si=std::sin(yaw),hx=b.size.x/2,hy=b.size.y/2;
      regions.push_back({b.center.x,b.center.y,b.center.z,hx,hy,b.size.z/2,co,si,
        std::abs(co)*hx+std::abs(si)*hy,std::abs(si)*hx+std::abs(co)*hy,
        b.class_name=="person" ? 0x00ff00ffU : 0x00ffa500U});
    }
    if(regions.empty())return out;
    const Received<Cloud> *selected=nullptr;
    for(auto it=point_ring_.rbegin();it!=point_ring_.rend();++it)
      if(fresh(*it) && same_header(it->message->header,header)){selected=&*it;break;}
    if(!selected)return out;
    const auto &c=*selected->message;
    const auto count=static_cast<std::uint64_t>(c.width)*c.height;
    if(count>300000 || c.point_step!=20 || c.fields.size()!=5 ||
       static_cast<std::uint64_t>(c.width)*c.point_step>c.row_step ||
       static_cast<std::uint64_t>(c.row_step)*c.height!=c.data.size())return out;
    const std::array<std::string,5> names{"x","y","z","rgb","source_index"};
    for(std::size_t i=0;i<names.size();++i) {
      const Field *f=nullptr;
      for(const auto &candidate:c.fields)if(candidate.name==names[i]){if(f)return out;f=&candidate;}
      if(!f || f->offset!=i*4 || f->count!=1 || f->datatype!=(i==4 ? Field::UINT32 : Field::FLOAT32))return out;
    }
    out.data.reserve(static_cast<std::size_t>(count)*20);
    for(std::uint32_t row=0;row<c.height;++row)for(std::uint32_t col=0;col<c.width;++col) {
      const auto *bytes=c.data.data()+static_cast<std::size_t>(row)*c.row_step+static_cast<std::size_t>(col)*c.point_step;
      const float x=read_scalar<float>(bytes,c.is_bigendian),y=read_scalar<float>(bytes+4,c.is_bigendian),z=read_scalar<float>(bytes+8,c.is_bigendian);
      if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))continue;
      for(const auto &b:regions) {
        const double dx=x-b.x,dy=y-b.y;
        if(std::abs(z-b.z)>b.hz || std::abs(dx)>b.bx || std::abs(dy)>b.by ||
           std::abs(b.co*dx+b.si*dy)>b.hx || std::abs(-b.si*dx+b.co*dy)>b.hy)continue;
        const auto start=out.data.size();out.data.resize(start+20);auto *destination=out.data.data()+start;
        write_scalar(destination,x);write_scalar(destination+4,y);write_scalar(destination+8,z);
        write_scalar(destination+12,b.rgb);write_scalar(destination+16,read_scalar<std::uint32_t>(bytes+16,c.is_bigendian));++out.width;break;
      }
    }
    out.row_step=out.width*20;return out;
  }
  void process() {
    const auto start=Clock::now();
    if(!fresh(lidar_)){lidar_input_valid_=false;clear("LiDAR detections stale or unavailable");return;}
    update_camera_validity();
    const bool camera_available=camera_latest_valid_ && !camera_ring_.empty() && fresh(camera_ring_.back());
    if(processed_lidar_==lidar_version_ && processed_camera_==camera_version_ && processed_info_==info_version_ && processed_points_==points_version_ &&
       last_camera_available_==camera_available)return;
    const bool new_lidar=processed_lidar_!=lidar_version_;
    if(!new_lidar && !lidar_input_valid_)return;
    processed_lidar_=lidar_version_;processed_camera_=camera_version_;processed_info_=info_version_;processed_points_=points_version_;
    last_camera_available_=camera_available;
    const auto &lidar=*lidar_.message;const auto stamp=stamp_ns(lidar.header);
    if(new_lidar && stamp<=last_lidar_stamp_){lidar_input_valid_=false;clear("Repeated/nonmonotonic LiDAR source timestamp");return;}
    if(new_lidar)last_lidar_stamp_=stamp;
    try {
      const auto boxes=decode_lidar(lidar);
      lidar_input_valid_=true;
      std::vector<CameraDetection> camera_boxes;
      const Received<CameraBoxes> *selected=nullptr;const Received<Info> *info=nullptr;
      double best_delta=pair_delta_+1;
      camera_status_="Camera unavailable, invalid or stale; LiDAR boxes retained";
      if(camera_available) {
        for(const auto &candidate:camera_ring_) {
          if(!fresh(candidate) || candidate.message->header.frame_id!=camera_frame_ || stamp_ns(candidate.message->header)<last_empty_camera_stamp_)continue;
          const double delta=std::abs(static_cast<double>(stamp_ns(candidate.message->header)-stamp))/1e9;
          if(delta>pair_delta_ || delta>=best_delta)continue;
          const Received<Info> *matching=nullptr;
          for(auto it=infos_.rbegin();it!=infos_.rend();++it)
            if(fresh(*it) && same_header(it->message->header,candidate.message->header) && valid_info(*it->message)){matching=&*it;break;}
          if(!matching)continue;
          selected=&candidate;info=matching;best_delta=delta;
        }
        if(selected) {
          try {camera_boxes=decode_camera(*selected->message);camera_status_=camera_boxes.empty() ? "Fresh empty camera evidence" : "Fresh time-aligned camera associations";}
          catch(const std::exception &e){camera_boxes.clear();selected=nullptr;info=nullptr;camera_status_=e.what();}
        } else camera_status_="No valid camera/CameraInfo pair within time window";
      }
      auto matches=associate_camera_boxes(boxes,camera_boxes,camera_,camera_from_sensor_,config_);
      auto markers=render(lidar,boxes,camera_boxes,matches);
      auto points=filtered_points(boxes,matches,lidar.header);
      if(!fresh(lidar_)){clear("LiDAR became stale during camera association");return;}
      if(selected && (!fresh(*selected) || !fresh(*info))) {
        camera_boxes.clear();matches.assign(boxes.size(),BoxCameraMatch{});
        markers=render(lidar,boxes,camera_boxes,matches);selected=nullptr;
        points=filtered_points(boxes,matches,lidar.header);
        camera_status_="Camera evidence became stale during association";
      }
      last_matches_=static_cast<std::size_t>(std::count_if(matches.begin(),matches.end(),[](const auto &m){return m.camera_index>=0;}));
      Lidar displayed;displayed.header=lidar.header;
      for(std::size_t i=0;i<boxes.size();++i)if(!require_camera_match_ || matches[i].camera_index>=0)displayed.detections.push_back(lidar.detections[i]);
      last_displayed_=displayed.detections.size();last_people_=static_cast<std::size_t>(std::count_if(displayed.detections.begin(),displayed.detections.end(),[](const auto &d){return d.results[0].hypothesis.class_id=="person";}));
      if(!fresh(lidar_)){lidar_input_valid_=false;clear("LiDAR became stale during filtered output preparation");return;}
      last_boxes_=boxes.size();last_delta_=selected ? best_delta : -1;
      last_camera_stamp_used_=selected ? stamp_ns(selected->message->header) : 0;
      markers_pub_->publish(std::move(markers));points_pub_->publish(std::move(points));detections_pub_->publish(std::move(displayed));++published_;cleared_=false;error_.clear();
      processing_ms_=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
      source_age_=static_cast<double>(now().nanoseconds()-stamp)/1e9;
    }catch(const std::exception &e){lidar_input_valid_=false;clear(e.what());}
  }
  void diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray m;m.header.stamp=now();
    Status s;s.name="camera_lidar_box_fusion";s.hardware_id="front_camera+ouster";s.level=Status::WARN;
    s.message=error_.empty() ? camera_status_ : error_;
    const auto add=[&s](const std::string &key,const std::string &value){diagnostic_msgs::msg::KeyValue v;v.key=key;v.value=value;s.values.push_back(v);};
    add("geometry_source","PointPillars_LiDAR");add("dimensions_modified","false");add("heading_modified","false");
    add("lidar_score_modified","false");add("joint_confidence_computed","false");add("camera_evidence","projected_box_association");
    add("physical_accuracy_validated","false");add("camera_calibration_validated","false");add("tf_published","false");
    add("output_frame","os_sensor");add("publishing_fresh_output",cleared_ ? "false" : "true");
    add("lidar_boxes",std::to_string(last_boxes_));add("camera_matches",std::to_string(last_matches_));
    add("camera_match_required",require_camera_match_ ? "true" : "false");add("displayed_boxes",std::to_string(last_displayed_));add("displayed_people",std::to_string(last_people_));
    add("published_updates",std::to_string(published_));add("cleared_outputs",std::to_string(clear_count_));
    add("processing_ms",std::to_string(processing_ms_));add("source_age_s",std::to_string(source_age_));
    add("header_delta_s",std::to_string(last_delta_));add("camera_header_ns",std::to_string(last_camera_stamp_used_));
    add("lidar_header_ns",std::to_string(last_lidar_stamp_));add("publication_cap_hz",std::to_string(rate_));
    m.status.push_back(std::move(s));diagnostics_pub_->publish(std::move(m));
  }
  Camera camera_;Transform camera_from_sensor_{};AssociationConfig config_;
  std::array<double,9> expected_r_{};std::array<double,12> expected_p_{};
  std::string camera_frame_,error_{"Waiting for LiDAR"},camera_status_{"Waiting for camera"};
  double pair_delta_{},stale_{},rate_{},processing_ms_{},source_age_{},last_delta_{-1};
  Received<Lidar> lidar_{};std::deque<Received<CameraBoxes>> camera_ring_;std::deque<Received<Info>> infos_;std::deque<Received<Cloud>> point_ring_;
  std::uint64_t lidar_version_{},camera_version_{},info_version_{},points_version_{},processed_lidar_{},processed_camera_{},processed_info_{},processed_points_{},checked_camera_{},published_{},clear_count_{};
  std::int64_t last_lidar_stamp_{},last_camera_stamp_{},last_empty_camera_stamp_{},last_camera_stamp_used_{};
  std::size_t last_boxes_{},last_matches_{},last_displayed_{},last_people_{};bool camera_latest_valid_{false},last_camera_available_{false},cleared_{false},lidar_input_valid_{false},require_camera_match_{false};
  rclcpp::Publisher<Markers>::SharedPtr markers_pub_;
  rclcpp::Publisher<Cloud>::SharedPtr points_pub_;
  rclcpp::Publisher<Lidar>::SharedPtr detections_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<Lidar>::SharedPtr lidar_sub_;
  rclcpp::Subscription<CameraBoxes>::SharedPtr camera_sub_;
  rclcpp::Subscription<Info>::SharedPtr info_sub_;
  rclcpp::Subscription<Cloud>::SharedPtr points_sub_;
  rclcpp::TimerBase::SharedPtr timer_,diag_timer_;
};
}  // namespace bev_visualization
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try{rclcpp::spin(std::make_shared<bev_visualization::CameraLidarBoxFusion>());}
  catch(const std::exception &e){fprintf(stderr,"camera_lidar_box_fusion: %s\n",e.what());rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
