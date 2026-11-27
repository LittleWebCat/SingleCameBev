#include "bev_visualization/camera_3d_geometry.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bev_visualization {
namespace {
using Clock = std::chrono::steady_clock;
using Image = sensor_msgs::msg::Image;
using Info = sensor_msgs::msg::CameraInfo;
using Detections = vision_msgs::msg::Detection3DArray;
using Markers = visualization_msgs::msg::MarkerArray;
constexpr int input_width = 1280, input_height = 384;
constexpr int feature_width = 320, feature_height = 96;
constexpr std::size_t input_elements = 3*input_width*input_height;
constexpr std::size_t heatmap_elements = 3*feature_width*feature_height;
constexpr std::size_t regression_elements = 8*feature_width*feature_height;
const std::array<std::string,3> classes{{"car", "bicycle", "person"}};
std::int64_t stamp_ns(const std_msgs::msg::Header &h) {
  return static_cast<std::int64_t>(h.stamp.sec)*1000000000LL+h.stamp.nanosec;
}
bool same_header(const std_msgs::msg::Header &a, const std_msgs::msg::Header &b) {
  return a.frame_id==b.frame_id && a.stamp.sec==b.stamp.sec && a.stamp.nanosec==b.stamp.nanosec;
}
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void cuda_check(cudaError_t result, const char *operation) {
  if(result!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *message) noexcept override {
    if(severity<=Severity::kERROR){try{error=message ? message : "TensorRT error";}catch(...){}}
  }
  std::string error;
};
struct Buffer {
  void *device{};float *host{};std::size_t count{};
  explicit Buffer(std::size_t n):count(n) {
    cuda_check(cudaMalloc(&device,n*sizeof(float)),"cudaMalloc");
    const auto result=cudaMallocHost(reinterpret_cast<void **>(&host),n*sizeof(float));
    if(result!=cudaSuccess){cudaFree(device);device=nullptr;cuda_check(result,"cudaMallocHost");}
  }
  ~Buffer(){if(device)cudaFree(device);if(host)cudaFreeHost(host);}
  Buffer(const Buffer &)=delete;
};
struct Stream {
  cudaStream_t value{};
  Stream(){cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"cudaStreamCreate");}
  ~Stream(){if(value)cudaStreamDestroy(value);}
};
struct Event {
  cudaEvent_t value{};
  Event(){cuda_check(cudaEventCreate(&value),"cudaEventCreate");}
  ~Event(){if(value)cudaEventDestroy(value);}
};
bool shape_is(const nvinfer1::Dims &d,std::initializer_list<int> expected) {
  if(d.nbDims!=static_cast<int>(expected.size()))return false;
  int i=0;for(const int n:expected)if(d.d[i++]!=n)return false;
  return true;
}
class SmokeEngine {
 public:
  explicit SmokeEngine(const std::string &path) {
    if(!initLibNvInferPlugins(&logger_,""))throw std::runtime_error("TensorRT plugins unavailable");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("SMOKE engine is missing: "+path);
    const auto size=file.tellg();
    if(size<=0 || size>1024LL*1024*1024)throw std::runtime_error("Invalid SMOKE engine size");
    std::vector<char> bytes(static_cast<std::size_t>(size));file.seekg(0);file.read(bytes.data(),size);
    if(!file)throw std::runtime_error("Cannot read SMOKE engine");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if(!runtime_)throw std::runtime_error("Cannot create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(),bytes.size()));
    if(!engine_)throw std::runtime_error("Cannot deserialize SMOKE: "+logger_.error);
    if(engine_->getNbIOTensors()!=3)throw std::runtime_error("SMOKE engine must have exactly three tensors");
    validate("image",nvinfer1::TensorIOMode::kINPUT,{1,3,384,1280});
    validate("heatmap",nvinfer1::TensorIOMode::kOUTPUT,{1,3,96,320});
    validate("regression",nvinfer1::TensorIOMode::kOUTPUT,{1,8,96,320});
    context_.reset(engine_->createExecutionContext());
    if(!context_)throw std::runtime_error("Cannot create SMOKE context");
    if(!context_->setTensorAddress("image",input_.device) ||
       !context_->setTensorAddress("heatmap",heatmap_.device) ||
       !context_->setTensorAddress("regression",regression_.device))
      throw std::runtime_error("Cannot assign SMOKE tensor buffers");
  }
  void prepare(const cv::Mat &rectified) {
    const double scale=static_cast<double>(input_width)/rectified.cols;
    const double shift=(input_height-rectified.rows*scale)/2;
    const cv::Matx23d affine(scale,0,0,0,scale,shift);
    cv::Mat cropped;cv::warpAffine(rectified,cropped,affine,{input_width,input_height},cv::INTER_LINEAR);
    constexpr std::array<float,3> mean{{.485F,.456F,.406F}},stdev{{.229F,.224F,.225F}};
    constexpr std::size_t plane=input_width*input_height;
    // The official SMOKE checkpoint normalizes BGR (not RGB) in this order.
    for(int y=0;y<input_height;++y){const auto *row=cropped.ptr<cv::Vec3b>(y);
      for(int x=0;x<input_width;++x){const auto i=static_cast<std::size_t>(y)*input_width+x;
        for(std::size_t c=0;c<3;++c)input_.host[c*plane+i]=(row[x][c]/255.F-mean[c])/stdev[c];}}
  }
  void infer(std::vector<float> &heatmap,std::vector<float> &regression) {
    cuda_check(cudaMemcpyAsync(input_.device,input_.host,input_elements*sizeof(float),cudaMemcpyHostToDevice,stream_.value),"SMOKE upload");
    cuda_check(cudaEventRecord(begin_.value,stream_.value),"CUDA start event");
    if(!context_->enqueueV3(stream_.value))throw std::runtime_error("SMOKE inference failed: "+logger_.error);
    cuda_check(cudaEventRecord(end_.value,stream_.value),"CUDA end event");
    cuda_check(cudaMemcpyAsync(heatmap_.host,heatmap_.device,heatmap_elements*sizeof(float),cudaMemcpyDeviceToHost,stream_.value),"SMOKE heatmap download");
    cuda_check(cudaMemcpyAsync(regression_.host,regression_.device,regression_elements*sizeof(float),cudaMemcpyDeviceToHost,stream_.value),"SMOKE regression download");
    cuda_check(cudaStreamSynchronize(stream_.value),"SMOKE synchronize");
    float ms=0;cuda_check(cudaEventElapsedTime(&ms,begin_.value,end_.value),"CUDA timing");gpu_ms=ms;
    heatmap.assign(heatmap_.host,heatmap_.host+heatmap_elements);
    regression.assign(regression_.host,regression_.host+regression_elements);
  }
  double gpu_ms{};
 private:
  void validate(const char *name,nvinfer1::TensorIOMode mode,std::initializer_list<int> shape) {
    if(engine_->getTensorIOMode(name)!=mode || engine_->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
       !shape_is(engine_->getTensorShape(name),shape))throw std::runtime_error(std::string("SMOKE tensor contract mismatch: ")+name);
  }
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  Stream stream_;Event begin_,end_;
  Buffer input_{input_elements},heatmap_{heatmap_elements},regression_{regression_elements};
};
cv::Scalar color(int id){return id==2 ? cv::Scalar(255,0,255) : id==1 ? cv::Scalar(255,255,0) : cv::Scalar(0,165,255);}
geometry_msgs::msg::Point bev_point(const cv::Vec3d &p,double z=.15) {
  geometry_msgs::msg::Point out;out.x=p[2];out.y=-p[0];out.z=z;return out;
}
}  // namespace

class Camera3DDetector final : public rclcpp::Node {
 public:
  Camera3DDetector():Node("camera_3d_detector") {
    const auto engine_path=declare_parameter<std::string>("engine_file","");
    camera_frame_=declare_parameter<std::string>("camera_frame","camera_front_optical_frame");
    bev_frame_=declare_parameter<std::string>("bev_frame","camera_front_bev");
    width_=declare_parameter<int>("width",1024);height_=declare_parameter<int>("height",768);
    rate_=declare_parameter<double>("max_publish_hz",20.0);stale_=declare_parameter<double>("stale_after_s",.5);
    threshold_=declare_parameter<double>("score_threshold",.25);
    const auto requested=declare_parameter<std::vector<std::string>>("classes",{"car","person"});
    for(const auto &name:requested){const auto found=std::find(classes.begin(),classes.end(),name);
      if(found==classes.end() || !allowed_.insert(static_cast<int>(found-classes.begin())).second)
        throw std::invalid_argument("SMOKE classes must be unique car/person/bicycle names");}
    if(engine_path.empty() || camera_frame_.empty() || bev_frame_.empty() || camera_frame_==bev_frame_ ||
       width_<=0 || height_<=0 || width_>4096 || height_>4096 || allowed_.empty() ||
       !std::isfinite(rate_) || rate_<=0 || rate_>20 || !std::isfinite(stale_) || stale_<=0 || stale_>1 ||
       !std::isfinite(threshold_) || threshold_<=0 || threshold_>1)throw std::invalid_argument("Invalid camera 3D parameters");
    engine_=std::make_unique<SmokeEngine>(engine_path);
    const auto image_topic=declare_parameter<std::string>("image_topic","/camera/front/image_raw");
    const auto info_topic=declare_parameter<std::string>("camera_info_topic","/camera/front/camera_info");
    detections_pub_=create_publisher<Detections>(declare_parameter<std::string>("detections_topic","/camera_lidar_bev/camera_detections_3d"),rclcpp::QoS(1).reliable());
    image_pub_=create_publisher<Image>(declare_parameter<std::string>("labeled_image_topic","/camera_lidar_bev/camera_3d_labeled"),rclcpp::SensorDataQoS().keep_last(1));
    markers_pub_=create_publisher<Markers>(declare_parameter<std::string>("markers_topic","/camera_lidar_bev/camera_3d_markers"),rclcpp::QoS(1).reliable());
    diagnostics_pub_=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics",rclcpp::QoS(5).reliable());
    tf_pub_=create_publisher<tf2_msgs::msg::TFMessage>("/tf_static",rclcpp::QoS(1).transient_local().reliable());
    image_sub_=create_subscription<Image>(image_topic,rclcpp::SensorDataQoS().keep_last(1),[this](Image::ConstSharedPtr msg){
      ++received_;if(image_ && version_!=processed_version_)++dropped_;image_=std::move(msg);receipt_=Clock::now();++version_;});
    info_sub_=create_subscription<Info>(info_topic,rclcpp::SensorDataQoS().keep_last(8),[this](Info::ConstSharedPtr msg){
      infos_.push_back(std::move(msg));while(infos_.size()>8)infos_.pop_front();});
    publish_axes();
    timer_=create_wall_timer(std::chrono::duration<double>(1/rate_),[this]{process();});
    diagnostic_timer_=create_wall_timer(std::chrono::seconds(1),[this]{diagnostics();});
    RCLCPP_INFO(get_logger(),"Camera-only trained SMOKE 3D: 1280x384 center crop, %.1f Hz cap; dimensions are predictions",rate_);
  }
 private:
  bool fresh(const Image &msg) const {
    const auto age=static_cast<double>(now().nanoseconds()-stamp_ns(msg.header))/1e9;
    const auto received_age=std::chrono::duration<double>(Clock::now()-receipt_).count();
    return msg.header.stamp.nanosec<1000000000U && stamp_ns(msg.header)>0 && age>=0 && age<=stale_ && received_age<=stale_;
  }
  void publish_axes() {
    // Pure coordinate convention, with the same camera origin and no vehicle extrinsic.
    // optical <- BEV: optical X=-BEV Y, optical Y=-BEV Z, optical Z=BEV X.
    geometry_msgs::msg::TransformStamped t;t.header.stamp=now();t.header.frame_id=camera_frame_;t.child_frame_id=bev_frame_;
    t.transform.rotation.x=.5;t.transform.rotation.y=-.5;t.transform.rotation.z=.5;t.transform.rotation.w=.5;
    tf2_msgs::msg::TFMessage out;out.transforms.push_back(t);tf_pub_->publish(out);
  }
  void rectify(const Info &info) {
    if(info.header.frame_id!=camera_frame_ || info.width!=static_cast<unsigned>(width_) || info.height!=static_cast<unsigned>(height_) ||
       info.distortion_model!="plumb_bob" || info.d.size()!=5 || info.k[0]<=0 || info.k[4]<=0)
      throw std::invalid_argument("CameraInfo must match native calibrated plumb_bob input");
    for(double v:info.k)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite camera intrinsics");
    for(double v:info.d)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite camera distortion");
    const std::array<double,9> identity{{1,0,0,0,1,0,0,0,1}};
    for(std::size_t i=0;i<9;++i)if(!std::isfinite(info.r[i]) || std::abs(info.r[i]-identity[i])>1e-8)throw std::invalid_argument("Raw camera requires finite identity R");
    if(std::abs(info.k[6])>1e-8 || std::abs(info.k[7])>1e-8 || std::abs(info.k[8]-1)>1e-8 ||
       info.k[2]<0 || info.k[2]>=width_ || info.k[5]<0 || info.k[5]>=height_ ||
       info.binning_x!=0 || info.binning_y!=0 || info.roi.x_offset!=0 || info.roi.y_offset!=0 ||
       info.roi.width!=0 || info.roi.height!=0 || info.roi.do_rectify)
      throw std::invalid_argument("CameraInfo must describe the full native pinhole image");
    for(std::size_t i=0;i<12;++i){const double expected=i%4==3 ? 0 : info.k[(i/4)*3+i%4];
      if(!std::isfinite(info.p[i]) || std::abs(info.p[i]-expected)>1e-6)throw std::invalid_argument("Raw camera requires P=[K|0]");}
    if(!map_x_.empty() && info.k==last_k_ && info.d==last_d_)return;
    std::copy(info.k.begin(),info.k.end(),k_.val);last_k_=info.k;last_d_=info.d;
    cv::initUndistortRectifyMap(cv::Mat(k_),cv::Mat(info.d),cv::Mat(),cv::Mat(k_),{width_,height_},CV_32FC1,map_x_,map_y_);
  }
  void publish_image(const cv::Mat &bgr,const std_msgs::msg::Header &header) {
    Image out;out.header=header;out.width=bgr.cols;out.height=bgr.rows;out.encoding="bgr8";out.is_bigendian=false;
    out.step=static_cast<std::uint32_t>(bgr.cols*3);out.data.resize(static_cast<std::size_t>(out.step)*out.height);
    for(int y=0;y<bgr.rows;++y)std::memcpy(out.data.data()+static_cast<std::size_t>(y)*out.step,bgr.ptr(y),out.step);
    image_pub_->publish(std::move(out));
  }
  void clear(const std::string &reason) {
    error_=reason;boxes_=0;if(cleared_)return;
    std_msgs::msg::Header header;header.stamp=now();header.frame_id=camera_frame_;
    Detections detections;detections.header=header;detections_pub_->publish(detections);
    visualization_msgs::msg::Marker m;m.header=header;m.header.frame_id=bev_frame_;m.action=m.DELETEALL;
    Markers markers;markers.markers.push_back(m);markers_pub_->publish(markers);
    cv::Mat blank(height_,width_,CV_8UC3,cv::Scalar());
    cv::putText(blank,"Camera 3D: waiting for fresh input",{20,40},cv::FONT_HERSHEY_SIMPLEX,.7,{255,255,255},2);
    publish_image(blank,header);cleared_=true;++clears_;
  }
  void draw(const std::vector<Camera3DBox> &boxes,cv::Mat &canvas,const std_msgs::msg::Header &header) {
    Detections detections;detections.header=header;Markers markers;
    visualization_msgs::msg::Marker clear_marker;clear_marker.header=header;clear_marker.header.frame_id=bev_frame_;clear_marker.action=clear_marker.DELETEALL;
    markers.markers.push_back(clear_marker);int id=0;
    for(const auto &box:boxes){
      const auto projected=project_camera_box(box,k_);if(!projected)continue;
      vision_msgs::msg::Detection3D d;d.header=header;
      d.bbox.center.position.x=box.center_xyz[0];d.bbox.center.position.y=box.center_xyz[1];d.bbox.center.position.z=box.center_xyz[2];
      d.bbox.center.orientation.y=std::sin(box.rotation_y/2);d.bbox.center.orientation.w=std::cos(box.rotation_y/2);
      d.bbox.size.x=box.size_lhw[0];d.bbox.size.y=box.size_lhw[1];d.bbox.size.z=box.size_lhw[2];
      vision_msgs::msg::ObjectHypothesisWithPose result;result.hypothesis.class_id=classes[box.class_id];result.hypothesis.score=box.score;
      result.pose.pose=d.bbox.center;
      d.results.push_back(result);detections.detections.push_back(d);
      const auto col=color(box.class_id);double min_u=width_,min_v=height_;
      for(std::size_t i=0;i<8;++i){min_u=std::min(min_u,(*projected)[i].x);min_v=std::min(min_v,(*projected)[i].y);
        for(std::size_t bit:{1U,2U,4U}){const auto j=i^bit;if(j<=i)continue;
          cv::Point a(static_cast<int>(std::clamp((*projected)[i].x,-10000.,10000.)),static_cast<int>(std::clamp((*projected)[i].y,-10000.,10000.)));
          cv::Point b(static_cast<int>(std::clamp((*projected)[j].x,-10000.,10000.)),static_cast<int>(std::clamp((*projected)[j].y,-10000.,10000.)));
          if(cv::clipLine(canvas.size(),a,b))cv::line(canvas,a,b,col,2,cv::LINE_AA);}}
      std::ostringstream text;text<<classes[box.class_id]<<" "<<std::fixed<<std::setprecision(2)<<box.score
        <<"  L "<<std::setprecision(1)<<box.size_lhw[0]<<" W "<<box.size_lhw[2]<<" H "<<box.size_lhw[1]<<" m";
      cv::putText(canvas,text.str(),{std::clamp(static_cast<int>(min_u),0,width_-1),std::clamp(static_cast<int>(min_v)-7,20,height_-1)},cv::FONT_HERSHEY_SIMPLEX,.48,col,1,cv::LINE_AA);
      const auto corners=camera_box_corners(box);std::vector<cv::Point2f> flat;
      for(const auto &corner:corners)flat.emplace_back(static_cast<float>(corner[2]),static_cast<float>(-corner[0]));
      std::vector<cv::Point2f> hull;cv::convexHull(flat,hull);
      visualization_msgs::msg::Marker line;line.header=header;line.header.frame_id=bev_frame_;line.ns="camera_3d_footprints";line.id=id++;
      line.type=line.LINE_STRIP;line.action=line.ADD;line.pose.orientation.w=1;line.scale.x=.07;
      line.color.r=static_cast<float>(col[2]/255);line.color.g=static_cast<float>(col[1]/255);line.color.b=static_cast<float>(col[0]/255);line.color.a=1;
      for(const auto &p:hull){geometry_msgs::msg::Point point;point.x=p.x;point.y=p.y;point.z=.15;line.points.push_back(point);}
      if(!line.points.empty())line.points.push_back(line.points.front());
      markers.markers.push_back(line);
      visualization_msgs::msg::Marker label=line;label.ns="camera_3d_labels";label.type=label.TEXT_VIEW_FACING;label.points.clear();
      label.pose.position=bev_point(box.center_xyz,.3);label.scale.x=0;label.scale.y=0;label.scale.z=.45;label.text=text.str();markers.markers.push_back(label);
    }
    boxes_=detections.detections.size();
    cv::putText(canvas,"Camera-only 3D predictions",{15,30},cv::FONT_HERSHEY_SIMPLEX,.65,{255,255,255},2,cv::LINE_AA);
    detections_pub_->publish(std::move(detections));markers_pub_->publish(std::move(markers));publish_image(canvas,header);
  }
  void process() {
    if(!image_){clear("Waiting for camera");return;}
    if(!fresh(*image_)){clear("Camera input is stale or has invalid timing");return;}
    if(version_==processed_version_)return;
    const auto image=image_;const auto stamp=stamp_ns(image->header);
    if(stamp<=last_stamp_){processed_version_=version_;clear("Repeated/nonmonotonic image timestamp");return;}
    Info::ConstSharedPtr info;
    for(auto it=infos_.rbegin();it!=infos_.rend();++it)if(same_header((*it)->header,image->header)){info=*it;break;}
    if(!info){clear("Waiting for paired CameraInfo");return;}
    processed_version_=version_;last_stamp_=stamp;const auto start=Clock::now();
    try{
      if(image->header.frame_id!=camera_frame_ || image->width!=static_cast<unsigned>(width_) || image->height!=static_cast<unsigned>(height_) ||
         (image->encoding!="bgr8" && image->encoding!="rgb8") || image->step<image->width*3 ||
         static_cast<std::uint64_t>(image->step)*image->height!=image->data.size())throw std::invalid_argument("Invalid native camera image");
      rectify(*info);
      cv::Mat raw(height_,width_,CV_8UC3,const_cast<unsigned char *>(image->data.data()),image->step),bgr;
      if(image->encoding=="rgb8")cv::cvtColor(raw,bgr,cv::COLOR_RGB2BGR);else bgr=raw;
      cv::Mat rectified;cv::remap(bgr,rectified,map_x_,map_y_,cv::INTER_LINEAR);
      engine_->prepare(rectified);std::vector<float> heatmap,regression;engine_->infer(heatmap,regression);
      auto boxes=decode_smoke(heatmap,regression,feature_width,feature_height,k_,width_,height_,threshold_,50);
      boxes.erase(std::remove_if(boxes.begin(),boxes.end(),[this](const auto &box){
        return !allowed_.count(box.class_id) || box.center_xyz[2]>60 || box.center_xyz[2]<.5 || std::abs(box.center_xyz[0])>40;
      }),boxes.end());
      if(!fresh(*image)){clear("Camera became stale during 3D inference");return;}
      draw(boxes,rectified,image->header);++processed_;cleared_=false;error_.clear();processing_ms_=elapsed_ms(start);
      source_age_=static_cast<double>(now().nanoseconds()-stamp)/1e9;
    }catch(const std::exception &e){++rejected_;clear(e.what());}
  }
  void diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray out;out.header.stamp=now();diagnostic_msgs::msg::DiagnosticStatus s;
    s.name="camera_3d_detector";s.hardware_id="front_camera";s.level=s.WARN;
    s.message=error_.empty() ? "Camera-only KITTI 3D predictions; physical accuracy unvalidated" : error_;
    const auto add=[&s](const std::string &key,const std::string &value){diagnostic_msgs::msg::KeyValue kv;kv.key=key;kv.value=value;s.values.push_back(kv);};
    add("model","SMOKE_DLA34_GN_DCN_KITTI");add("camera_only","true");add("trained_3d_detector","true");
    add("physical_accuracy_validated","false");add("lidar_used","false");add("tracking_enabled","false");
    add("camera_frame",camera_frame_);add("bev_frame",bev_frame_);add("publishing_fresh_output",cleared_ ? "false" : "true");
    add("received_frames",std::to_string(received_));add("processed_frames",std::to_string(processed_));add("mailbox_dropped_frames",std::to_string(dropped_));
    add("rejected_frames",std::to_string(rejected_));add("cleared_outputs",std::to_string(clears_));add("boxes",std::to_string(boxes_));
    add("inference_ms",std::to_string(engine_->gpu_ms));add("processing_ms",std::to_string(processing_ms_));add("source_age_s",std::to_string(source_age_));
    add("publication_cap_hz",std::to_string(rate_));out.status.push_back(std::move(s));diagnostics_pub_->publish(out);
  }
  std::unique_ptr<SmokeEngine> engine_;
  std::string camera_frame_,bev_frame_,error_;int width_{},height_{};
  double rate_{},stale_{},threshold_{},processing_ms_{},source_age_{};std::set<int> allowed_;
  cv::Mat map_x_,map_y_;cv::Matx33d k_;std::array<double,9> last_k_{};std::vector<double> last_d_;
  Image::ConstSharedPtr image_;Clock::time_point receipt_;std::deque<Info::ConstSharedPtr> infos_;
  std::uint64_t received_{},processed_{},dropped_{},rejected_{},clears_{},version_{},processed_version_{};std::int64_t last_stamp_{};
  std::size_t boxes_{};bool cleared_{false};
  rclcpp::Publisher<Detections>::SharedPtr detections_pub_;rclcpp::Publisher<Image>::SharedPtr image_pub_;
  rclcpp::Publisher<Markers>::SharedPtr markers_pub_;rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;rclcpp::Subscription<Info>::SharedPtr info_sub_;
  rclcpp::TimerBase::SharedPtr timer_,diagnostic_timer_;
};
}  // namespace bev_visualization
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  try{rclcpp::spin(std::make_shared<bev_visualization::Camera3DDetector>());}
  catch(const std::exception &e){std::cerr<<"camera_3d_detector: "<<e.what()<<'\n';rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
