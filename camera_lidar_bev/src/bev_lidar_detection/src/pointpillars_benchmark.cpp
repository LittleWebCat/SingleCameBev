#include "bev_lidar_detection/detector.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
  try {
    if(argc!=3) throw std::runtime_error("Usage: pointpillars_benchmark engine.plan scan.bin");
    std::ifstream f(argv[2],std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("Cannot open reference scan");
    const auto n=f.tellg();
    if(n<=0 || n%sizeof(bev_lidar_detection::PointXYZI)!=0) throw std::runtime_error("Invalid XYZI reference scan");
    std::vector<bev_lidar_detection::PointXYZI> points(n/sizeof(bev_lidar_detection::PointXYZI));
    f.seekg(0); f.read(reinterpret_cast<char *>(points.data()),n);
    bev_lidar_detection::Detector detector(argv[1]);
    std::vector<double> timings;
    std::vector<bev_lidar_detection::Box3D> boxes;
    for(int i=0;i<35;++i) {
      boxes=detector.infer(points);
      if(i>=5)timings.push_back(detector.lastStats().processing_ms);
    }
    std::sort(timings.begin(),timings.end());
    const auto stats=detector.lastStats();
    std::cout<<std::setprecision(8)<<"{\"input_points\":"<<stats.input_points<<",\"accepted_points\":"<<stats.accepted_points
      <<",\"input_pillars\":"<<stats.input_pillars<<",\"accepted_pillars\":"<<stats.accepted_pillars
      <<",\"truncated_pillars\":"<<stats.truncated_pillars<<",\"warmup\":5,\"runs\":30,\"median_ms\":"<<timings[timings.size()/2]
      <<",\"p95_ms\":"<<timings[static_cast<std::size_t>(std::ceil(.95*timings.size()))-1]
      <<",\"max_ms\":"<<timings.back()<<",\"boxes\":[";
    for(std::size_t i=0;i<boxes.size();++i) {
      const auto &b=boxes[i]; if(i)std::cout<<',';
      std::cout<<"{\"class_id\":"<<b.class_id<<",\"score\":"<<b.score<<",\"center\":["<<b.x<<','<<b.y<<','<<b.z
        <<"],\"size\":["<<b.size_x<<','<<b.size_y<<','<<b.size_z<<"],\"yaw\":"<<b.yaw<<'}';
    }
    std::cout<<"]}\n";
    return boxes.empty()?2:0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
