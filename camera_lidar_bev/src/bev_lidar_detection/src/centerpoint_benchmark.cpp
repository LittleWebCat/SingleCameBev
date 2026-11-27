#include "bev_lidar_detection/centerpoint_detector.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Usage: centerpoint_benchmark engine.plan points_model_frame_f32x5.bin [z_offset] [score_threshold]
// The reference file holds nuScenes-frame (x right, y forward) points with 0..255 intensity, as
// written by tools/build_lidar_centerpoint_engine.py; they are converted back to os_sensor here.
int main(int argc,char **argv) {
  try {
    if(argc<3 || argc>5) throw std::runtime_error("Usage: centerpoint_benchmark engine.plan points_f32x5.bin [z_offset] [score_threshold]");
    const float z_offset=argc>3 ? std::stof(argv[3]) : 0.0f;
    std::ifstream f(argv[2],std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("Cannot open reference points");
    const auto n=static_cast<std::size_t>(f.tellg());
    if(n==0 || n%(5*sizeof(float))!=0) throw std::runtime_error("Invalid f32x5 reference points");
    std::vector<float> raw(n/sizeof(float));
    f.seekg(0); f.read(reinterpret_cast<char *>(raw.data()),static_cast<std::streamsize>(n));
    std::vector<bev_lidar_detection::PointXYZI> points;
    for(std::size_t i=0;i<raw.size();i+=5) points.push_back({raw[i+1],-raw[i],raw[i+2]-z_offset,raw[i+3]/255.0f});
    bev_lidar_detection::CenterPointConfig config;
    config.engine_path=argv[1];config.z_offset=z_offset;
    if(argc>4) config.score_threshold=std::stof(argv[4]);
    bev_lidar_detection::CenterPointDetector detector(config);
    std::vector<double> total,pre,net,post;
    std::vector<bev_lidar_detection::Box3D> boxes;
    for(int i=0;i<55;++i) {
      boxes=detector.infer(points);
      if(i<5) continue;
      const auto &s=detector.lastStats();
      total.push_back(s.processing_ms);pre.push_back(s.preprocess_ms);net.push_back(s.inference_ms);post.push_back(s.postprocess_ms);
    }
    const auto med=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    const auto p95=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[static_cast<std::size_t>(std::ceil(.95*v.size()))-1];};
    const auto &s=detector.lastStats();
    const auto &names=detector.classNames();
    std::cout<<std::setprecision(8)<<"{\"model\":\""<<detector.modelName()<<"\",\"input_points\":"<<s.input_points
      <<",\"accepted_points\":"<<s.accepted_points<<",\"input_pillars\":"<<s.input_pillars
      <<",\"accepted_pillars\":"<<s.accepted_pillars<<",\"truncated_pillars\":"<<s.truncated_pillars
      <<",\"truncated_points\":"<<s.truncated_points<<",\"warmup\":5,\"runs\":50"
      <<",\"median_ms\":"<<med(total)<<",\"p95_ms\":"<<p95(total)
      <<",\"median_preprocess_ms\":"<<med(pre)<<",\"median_inference_ms\":"<<med(net)<<",\"median_postprocess_ms\":"<<med(post)
      <<",\"boxes\":[";
    for(std::size_t i=0;i<boxes.size();++i) {
      const auto &b=boxes[i]; if(i)std::cout<<',';
      std::cout<<"{\"class\":\""<<names.at(static_cast<std::size_t>(b.class_id))<<"\",\"score\":"<<b.score
        <<",\"center\":["<<b.x<<','<<b.y<<','<<b.z<<"],\"size\":["<<b.size_x<<','<<b.size_y<<','<<b.size_z<<"],\"yaw\":"<<b.yaw<<'}';
    }
    std::cout<<"]}\n";
    return boxes.empty()?2:0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
