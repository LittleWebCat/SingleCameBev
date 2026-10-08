#include "bev_lidar_detection/detector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "geometry_msgs/msg/point.hpp"
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
geometry_msgs::msg::Point point(double x,double y,double z) {
  geometry_msgs::msg::Point p; p.x=x;p.y=y;p.z=z;return p;
}
const char *class_name(int id) {return id==1 ? "person" : "car";}
std::uint32_t class_rgb(int id) {return id==1 ? 0x00ff00ff : 0x00ffa500;}
void class_color(Marker &m,int id) {m.color.r=1;m.color.g=id==1 ? 0 : 165.0F/255;m.color.b=id==1 ? 1 : 0;m.color.a=1;}
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
    const auto points=declare_parameter("points_topic","/camera_lidar_bev/lidar_vehicle_points");
    frame_=declare_parameter("output_frame","os_sensor");
    intensity_field_=declare_parameter("intensity_field","reflectivity");
    intensity_scale_=declare_parameter("intensity_scale",255.0);
    score_=declare_parameter("score_threshold",0.35);
    rate_=declare_parameter("max_publish_hz",20.0);
    stale_=declare_parameter("stale_after_s",0.5);
    max_forward_=declare_parameter("max_forward_m",60.0);
    const auto classes=declare_parameter<std::vector<std::string>>("classes",{"car","person"});
    enabled_classes_={false,false};
    for(const auto &name:classes) {
      const int id=name=="car" ? 0 : name=="person" ? 1 : -1;
      if(id<0 || enabled_classes_[id])throw std::invalid_argument("classes must contain unique car/person entries");
      enabled_classes_[id]=true;
    }
    if(classes.empty())throw std::invalid_argument("At least one output class is required");
    mask_=declare_parameter("self_mask_enabled",true);
    mask_min_={declare_parameter("self_mask_min_x_m",-2.131),declare_parameter("self_mask_min_y_m",-0.905),declare_parameter("self_mask_min_z_m",-1.8)};
    mask_max_={declare_parameter("self_mask_max_x_m",2.350),declare_parameter("self_mask_max_y_m",0.885),declare_parameter("self_mask_max_z_m",0.0)};
    if (engine.empty() || input.empty() || detections.empty() || markers.empty() || points.empty() ||
        frame_!="os_sensor" || intensity_field_.empty() || !std::isfinite(intensity_scale_) || intensity_scale_<=0 ||
        !std::isfinite(score_) || score_<=0 || score_>=1 || !std::isfinite(rate_) || rate_<=0 || rate_>20 ||
        !std::isfinite(stale_) || stale_<=0 || stale_>0.75 || !std::isfinite(max_forward_) || max_forward_<=0 || max_forward_>69.12)
      throw std::invalid_argument("Invalid trained LiDAR detector parameters");
    for (int i=0;i<3;++i) if (!std::isfinite(mask_min_[i]) || !std::isfinite(mask_max_[i]) || mask_min_[i]>mask_max_[i])
      throw std::invalid_argument("Invalid measured vehicle self mask");
    detector_=std::make_unique<Detector>(engine,static_cast<float>(score_),200);
    detections_pub_=create_publisher<Detections>(detections,rclcpp::QoS(1).reliable());
    markers_pub_=create_publisher<Markers>(markers,rclcpp::QoS(1).reliable());
    points_pub_=create_publisher<Cloud>(points,rclcpp::QoS(1).reliable());
    diagnostics_pub_=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics",rclcpp::QoS(5));
    cloud_sub_=create_subscription<Cloud>(input,rclcpp::SensorDataQoS().keep_last(1),
      [this](Cloud::ConstSharedPtr c){latest_=std::move(c);receipt_=Clock::now();++version_;++received_;});
    tf_sub_=create_subscription<tf2_msgs::msg::TFMessage>("/tf_static",rclcpp::QoS(10).reliable().transient_local(),
      [this](tf2_msgs::msg::TFMessage::ConstSharedPtr m){observe_transform(*m);});
    timer_=create_wall_timer(std::chrono::duration<double>(1.0/rate_),[this]{process();});
    diag_timer_=create_wall_timer(std::chrono::seconds(1),[this]{diagnostics();});
    RCLCPP_WARN(get_logger(),"PointPillars KITTI car/person predictions: complete dimensions/yaw; Ouster accuracy and reflectivity adaptation remain unvalidated. Flat BEV display, no vehicle TF or tracking.");
  }

 private:
  void observe_transform(const tf2_msgs::msg::TFMessage &message) {
    for (const auto &t : message.transforms) {
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
      markers_pub_->publish(empty_markers(h));points_pub_->publish(empty_points(h));++clear_count_;
    }
    cleared_=true;error_=reason;last_cars_=0;last_people_=0;last_point_count_=0;
  }
  void decode(const Cloud &c,std::vector<PointXYZI> &points,std::vector<std::uint32_t> &indices) {
    const auto count=static_cast<std::uint64_t>(c.width)*c.height;
    if (!count || count>Detector::max_points || !c.point_step || c.point_step>256 ||
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
    auto markers=empty_markers(h);auto cloud=empty_points(h);
    for (std::size_t i=0;i<boxes.size();++i) {
      const auto &b=boxes[i];
      vision_msgs::msg::Detection3D d;d.header=h;
      d.bbox.center.position=point(b.x,b.y,b.z);
      d.bbox.center.orientation.z=std::sin(b.yaw/2);d.bbox.center.orientation.w=std::cos(b.yaw/2);
      d.bbox.size.x=b.size_x;d.bbox.size.y=b.size_y;d.bbox.size.z=b.size_z;
      vision_msgs::msg::ObjectHypothesisWithPose result;result.hypothesis.class_id=class_name(b.class_id);result.hypothesis.score=b.score;
      result.pose.pose=d.bbox.center;d.results.push_back(std::move(result));detections.detections.push_back(std::move(d));
      Marker rectangle;rectangle.header=h;rectangle.ns="predicted_vehicle_bev_extent";rectangle.id=static_cast<int>(i*2);
      rectangle.type=Marker::LINE_STRIP;rectangle.action=Marker::ADD;rectangle.pose.orientation.w=1;
      rectangle.scale.x=0.08;rectangle.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(rectangle,b.class_id);
      const double co=std::cos(b.yaw),si=std::sin(b.yaw);
      for (const auto &sign : std::array<std::pair<int,int>,5>{{{-1,-1},{1,-1},{1,1},{-1,1},{-1,-1}}}) {
        const double dx=sign.first*b.size_x/2,dy=sign.second*b.size_y/2;
        rectangle.points.push_back(point(b.x+co*dx-si*dy,b.y+si*dx+co*dy,0.15));
      }
      markers.markers.push_back(std::move(rectangle));
      Marker label;label.header=h;label.ns="predicted_vehicle_bev_labels";label.id=static_cast<int>(i*2+1);
      label.type=Marker::TEXT_VIEW_FACING;label.action=Marker::ADD;label.pose.orientation.w=1;
      label.pose.position=point(b.x,b.y,0.17);label.scale.z=0.55;
      label.lifetime=rclcpp::Duration::from_seconds(0.5);class_color(label,b.class_id);
      std::ostringstream text;text<<class_name(b.class_id)<<' '<<std::fixed<<std::setprecision(2)<<b.score<<"\nL "<<b.size_x<<" W "<<b.size_y<<" H "<<b.size_z<<" m";
      label.text=text.str();markers.markers.push_back(std::move(label));
    }
    if (points_pub_->get_subscription_count()) {
      struct Region {double x,y,z,hx,hy,hz,co,si,bound_x,bound_y;int class_id;};
      std::vector<Region> regions;regions.reserve(boxes.size());
      for (const auto &b:boxes) {
        const double co=std::cos(b.yaw),si=std::sin(b.yaw),hx=b.size_x/2,hy=b.size_y/2;
        regions.push_back({b.x,b.y,b.z,hx,hy,b.size_z/2,co,si,std::abs(co)*hx+std::abs(si)*hy,std::abs(si)*hx+std::abs(co)*hy,b.class_id});
      }
      cloud.data.reserve(points.size()*cloud.point_step);
      for (std::size_t i=0;i<points.size();++i) {
        const auto &p=points[i];bool inside=false;std::uint32_t rgb=0;
        for (const auto &b:regions) {
          const double dx=p.x-b.x,dy=p.y-b.y;
          if (std::abs(p.z-b.z)>b.hz || std::abs(dx)>b.bound_x || std::abs(dy)>b.bound_y) continue;
          if (std::abs(b.co*dx+b.si*dy)<=b.hx && std::abs(-b.si*dx+b.co*dy)<=b.hy) {inside=true;rgb=class_rgb(b.class_id);break;}
        }
        if (!inside) continue;
        const auto start=cloud.data.size();cloud.data.resize(start+cloud.point_step);
        auto *bytes=cloud.data.data()+start;write(bytes,p.x);write(bytes+4,p.y);write(bytes+8,p.z);
        write(bytes+12,rgb);write(bytes+16,indices[i]);++cloud.width;
      }
    }
    cloud.row_step=cloud.width*cloud.point_step;last_point_count_=cloud.width;
    if (!fresh(source,receipt)) {clear("LiDAR scan became stale during output preparation");return false;}
    detections_pub_->publish(std::move(detections));markers_pub_->publish(std::move(markers));points_pub_->publish(std::move(cloud));
    return true;
  }
  void process() {
    if (tf_conflict_) {clear("Ouster driver TF conflicts with model coordinates");return;}
    if (!tf_valid_) {clear("Waiting for actual Ouster os_sensor <- os_lidar static TF");return;}
    if (!latest_ || !fresh(*latest_,receipt_)) {clear("LiDAR input stale or unavailable");return;}
    if (version_==processed_) return;
    const auto start=Clock::now();const auto cloud=latest_;const auto receipt=receipt_;processed_=version_;
    const auto stamp=stamp_ns(cloud->header);
    if (stamp<=last_stamp_) {clear("Repeated/nonmonotonic LiDAR source timestamp");return;}
    last_stamp_=stamp;
    try {
      std::vector<PointXYZI> points;std::vector<std::uint32_t> indices;decode(*cloud,points,indices);
      const auto predicted=detector_->infer(points);
      std::vector<Box3D> objects;
      for (const auto &b:predicted) {
        if (b.class_id<0 || b.class_id>1 || !enabled_classes_[b.class_id] || b.score<score_ || b.score>1 || b.x<0 || b.x>max_forward_ ||
            b.size_x>15 || b.size_y>6 || b.size_z>5 || masked(b.x,b.y,b.z)) continue;
        objects.push_back(b);
      }
      if (!fresh(*cloud,receipt)) {clear("LiDAR scan became stale during inference");return;}
      if (!publish(objects,points,indices,*cloud,receipt)) return;
      last_cars_=static_cast<std::size_t>(std::count_if(objects.begin(),objects.end(),[](const auto &b){return b.class_id==0;}));
      last_people_=objects.size()-last_cars_;++processed_count_;
      cleared_=false;error_.clear();last_age_=static_cast<double>(now().nanoseconds()-stamp)/1e9;
      processing_ms_=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    } catch (const std::exception &e) {clear(e.what());}
  }
  void diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message;message.header.stamp=now();
    Status s;s.name="lidar_vehicle_detector";s.hardware_id="ouster+Thor";s.level=Status::WARN;
    s.message=error_.empty() ? "Trained KITTI car/person predictions active; Ouster accuracy remains provisional" : error_;
    const auto add=[&s](const std::string &key,const std::string &value){diagnostic_msgs::msg::KeyValue v;v.key=key;v.value=value;s.values.push_back(v);};
    const auto &stats=detector_->lastStats();
    add("model","PointPillars_KITTI_FP16");add("trained_3d_detector","true");add("full_dimensions_predicted","true");
    add("heading_predicted","true");add("physical_accuracy_validated","false");add("tracking_enabled","false");
    add("frame",frame_);add("driver_tf_valid",tf_valid_ && !tf_conflict_ ? "true" : "false");
    add("publishing_fresh_output",cleared_ ? "false" : "true");
    add("vehicle_class",enabled_classes_[0] && enabled_classes_[1] ? "Car_and_Pedestrian" : enabled_classes_[0] ? "Car_only" : "Pedestrian_only");
    add("intensity_field",intensity_field_);add("intensity_scale",std::to_string(intensity_scale_));
    add("received_scans",std::to_string(received_));add("processed_scans",std::to_string(processed_count_));
    add("cleared_outputs",std::to_string(clear_count_));add("vehicle_boxes",std::to_string(last_cars_));
    add("person_boxes",std::to_string(last_people_));add("object_boxes",std::to_string(last_cars_+last_people_));
    add("vehicle_points",std::to_string(last_point_count_));add("self_masked_points",std::to_string(last_masked_));
    add("model_input_points",std::to_string(stats.accepted_points));add("detector_ms",std::to_string(stats.processing_ms));
    add("input_pillars",std::to_string(stats.input_pillars));add("accepted_pillars",std::to_string(stats.accepted_pillars));
    add("truncated_pillars",std::to_string(stats.truncated_pillars));
    add("processing_ms",std::to_string(processing_ms_));add("source_age_s",std::to_string(last_age_));
    add("publication_cap_hz",std::to_string(rate_));message.status.push_back(std::move(s));diagnostics_pub_->publish(std::move(message));
  }
  std::unique_ptr<Detector> detector_;
  std::string frame_,intensity_field_,error_{"Waiting for LiDAR"};
  double intensity_scale_{},score_{},rate_{},stale_{},max_forward_{},last_age_{},processing_ms_{};
  std::array<double,3> mask_min_{},mask_max_{};
  std::array<bool,2> enabled_classes_{};
  bool mask_{},tf_valid_{false},tf_conflict_{false},cleared_{false};
  Cloud::ConstSharedPtr latest_;Clock::time_point receipt_{};
  std::uint64_t version_{},processed_{},received_{},processed_count_{},clear_count_{};
  std::int64_t last_stamp_{0};std::size_t last_masked_{},last_cars_{},last_people_{},last_point_count_{};
  rclcpp::Publisher<Detections>::SharedPtr detections_pub_;
  rclcpp::Publisher<Markers>::SharedPtr markers_pub_;
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
