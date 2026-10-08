#include "bev_lidar_detection/detector.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv) {
  if(argc!=2)return 1;
  bev_lidar_detection::Detector detector(argv[1]);
  auto empty=detector.infer({});
  if(!empty.empty() || detector.lastStats().accepted_points!=0 || detector.lastStats().output_boxes!=0)return 2;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  std::vector<bev_lidar_detection::PointXYZI> invalid={
    {nan,0,-1,.1},{10,nan,-1,.1},{10,0,nan,.1},{10,0,-1,nan},
    {-1,0,-1,.1},{69.12,0,-1,.1},{10,-40,-1,.1},{10,40,-1,.1},
    {10,0,-4,.1},{10,0,1,.1},{10,0,-1,-.1},{10,0,-1,1.1}};
  if(!detector.infer(invalid).empty() || detector.lastStats().accepted_points!=0)return 3;
  std::vector<bev_lidar_detection::PointXYZI> oversized(bev_lidar_detection::Detector::max_points+1,{10,0,-1,.1});
  bool refused=false;
  try{detector.infer(oversized);}catch(const std::invalid_argument&){refused=true;}
  if(!refused)return 4;
  std::cout<<"{\"empty_input_clear\":true,\"nonfinite_out_of_range_invalid_intensity_rejected\":true,\"oversized_input_rejected\":true}\n";
}
