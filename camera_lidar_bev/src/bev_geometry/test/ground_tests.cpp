#include "bev_geometry/ground.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using bev_geometry::Config;
using bev_geometry::Label;
using bev_geometry::Point;
void require(bool condition,const std::string &message) {
  if (!condition) { throw std::runtime_error(message); }
}
template<class Function> void invalid(Function function,const std::string &message) {
  try { function(); } catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(message);
}
Config measured_config() { Config c; c.ground_height_m=1.8; return c; }
std::vector<Point> floor(double a=0.0,double b=0.0,double c=-1.8) {
  std::vector<Point> points;
  for (int x=5;x<=35;++x) {
    for (int y=-15;y<=15;++y) {
      const double px=x*.4+.03, py=y*.4+.03;
      const double noise=((x+y)%3)*.002;
      points.push_back({px,py,a*px+b*py+c+noise});
    }
  }
  return points;
}
void sloped_floor_box_and_dominant_wall() {
  auto points=floor(.04,-.02);
  const auto floor_count=points.size();
  const auto box_index=points.size();
  points.push_back({6.03,.03,.04*6.03-.02*.03-1.8+.8});
  const auto curb_index=points.size();
  points.push_back({5.23,1.23,.04*5.23-.02*1.23-1.8+.18});
  const auto drop_index=points.size();
  points.push_back({4.03,-1.57,.04*4.03-.02*(-1.57)-1.8-.3});
  // The wall greatly outnumbers the floor but occupies a narrow support strip.
  const auto wall_start=points.size();
  for (int y=-180;y<=180;++y) {
    for (int z=0;z<=30;++z) {
      const double py=y*.03;
      points.push_back({8.03,py,.04*8.03-.02*py-1.8+z*.06});
    }
  }
  const auto result=bev_geometry::classify_and_grid(points,measured_config());
  require(result.ground.valid,"sloped floor must survive a dominant vertical wall");
  require(std::abs(result.ground.a-.04)<.002 && std::abs(result.ground.b+.02)<.002,
          "floor slope fit must match synthetic geometry");
  require(result.ground.support_patches>=12 && result.ground.support_points>=80,
          "accepted ground must report broad support");
  require(result.ground.rms_m<.01,"floor fit residual must remain small");
  require(std::count(result.labels.begin(),result.labels.begin()+floor_count,Label::Ground)>
          static_cast<std::ptrdiff_t>(floor_count*.95),"observed floor returns must be ground");
  require(result.labels[box_index]==Label::Obstacle,"box must be obstacle");
  require(result.labels[curb_index]==Label::Obstacle,"18 cm curb must not become a local floor");
  require(result.labels[drop_index]==Label::Unknown,"below-floor return must remain unknown");
  require(std::abs(result.heights_above_ground[box_index]-.8)<.01,"reported height must use local floor");
  require(result.labels[wall_start+15]==Label::Obstacle,"wall above supported floor must be obstacle");
}
void reject_unsupported_planes() {
  auto c=measured_config();
  for (const auto &points : {std::vector<Point>{},floor(0,0,-.7),floor(.6,0,-1.8)}) {
    const auto result=bev_geometry::classify_and_grid(points,c);
    require(!result.ground.valid,"empty, wrong-height roof, and steep planes must be rejected");
    require(std::none_of(result.labels.begin(),result.labels.end(),[](Label l){return l==Label::Ground||l==Label::Obstacle;}),
            "rejected floor must not create classified geometry");
    require(std::all_of(result.grid.cells().begin(),result.grid.cells().end(),[](const bev_core::Cell &cell){
      return cell.occupancy==bev_core::Occupancy::Unknown;}),"rejected floor must leave unknown grid");
  }
  std::vector<Point> wall;
  for (int y=-100;y<=100;++y) {
    for (int z=0;z<40;++z) { wall.push_back({8,y*.05,-2.5+z*.08}); }
  }
  require(!bev_geometry::classify_and_grid(wall,c).ground.valid,"wall-only scan cannot be a floor");
  std::vector<Point> diagonal;
  for (int t=-100;t<=100;++t) {
    for (int z=0;z<20;++z) { diagonal.push_back({t*.05+6,t*.05,-1.8+z*.08}); }
  }
  require(!bev_geometry::classify_and_grid(diagonal,c).ground.valid,"long diagonal strip needs two-dimensional coverage");
  Config automatic;
  const auto low=bev_geometry::classify_and_grid(floor(),automatic);
  require(low.ground.valid && !low.ground.used_height_prior,"auto mode may identify a provisional broad low surface");
  require(!bev_geometry::classify_and_grid(floor(0,0,.4),automatic).ground.valid,"overhead plane cannot be auto ground");
}
void occlusion_and_unknown_gaps() {
  auto points=floor();
  const auto blocker=points.size();
  // y=.03 is inside one cell and blocks a distant collinear ground return.
  points.push_back({5.03,.03,-1.0});
  const auto behind=points.size();
  points.push_back({10.06,.06,-1.8});
  const auto result=bev_geometry::classify_and_grid(points,measured_config());
  const auto occupied=result.grid.index_for_point(points[blocker].x,points[blocker].y);
  const auto hidden=result.grid.index_for_point(points[behind].x,points[behind].y);
  require(occupied && result.grid.cell(*occupied).occupancy==bev_core::Occupancy::Occupied,
          "occupied cell must dominate coincident floor evidence");
  require(hidden && result.grid.cell(*hidden).occupancy==bev_core::Occupancy::Unknown,
          "ground observed over an obstacle must not clear space behind it");
  const auto blind=result.grid.index_for_point(.3,0);
  require(blind && result.grid.cell(*blind).occupancy==bev_core::Occupancy::Unknown,
          "unobserved near-field cannot be free by plane extrapolation");
  const auto sparse=result.grid.index_for_point(3.5,2.1);
  require(sparse && result.grid.cell(*sparse).occupancy==bev_core::Occupancy::Unknown,
          "sparse gaps between observed floor endpoints must stay unknown");
  const auto clear=result.grid.index_for_point(4.03,2.03);
  require(clear && result.grid.cell(*clear).occupancy==bev_core::Occupancy::Free,
          "direct unobstructed ground evidence should mark free");
}
void preserve_indices_and_explicit_masks() {
  auto points=floor();
  const auto start=points.size();
  points.push_back({std::numeric_limits<double>::quiet_NaN(),1,1});
  points.push_back({0,0,0});
  points.push_back({.1,.1,.1});
  points.push_back({100,0,-1.8});
  points.push_back({3.03,2.03,4});
  points.push_back({.7,.2,-.1});
  auto c=measured_config();
  const auto unmasked=bev_geometry::classify_and_grid(points,c);
  require(unmasked.labels.size()==points.size() && unmasked.heights_above_ground.size()==points.size(),
          "result vectors must preserve original input indices");
  require(unmasked.labels[start]==Label::Invalid && unmasked.labels[start+1]==Label::Invalid,
          "nonfinite and missing zero returns are invalid");
  require(unmasked.labels[start+2]==Label::Unknown && unmasked.labels[start+3]==Label::Unknown,
          "range limits cannot create ground or obstacles");
  require(unmasked.labels[start+4]==Label::Unknown,"overhead beyond obstacle height limit stays unknown");
  require(unmasked.labels.back()!=Label::Masked,"vehicle mask must be disabled by default");
  c.self_mask_enabled=true; c.self_x_max_m=1;
  const auto masked=bev_geometry::classify_and_grid(points,c);
  require(masked.labels.back()==Label::Masked && std::isnan(masked.heights_above_ground.back()),
          "explicit measured body box excludes points before fitting");
  require(masked.labels[start]==Label::Invalid,"self mask must not relabel invalid returns");
  require(!masked.grid.index_for_point(40,0) && !masked.grid.index_for_point(1,15),
          "grid maximum boundaries are half open");
}
void invalid_configuration() {
  auto c=measured_config(); c.ground_height_m=0;
  invalid([&]{bev_geometry::classify_and_grid({},c);},"height zero must be rejected");
  c=measured_config(); c.ground_distance_m=.2;
  invalid([&]{bev_geometry::validate_config(c);},"overlapping ground/obstacle thresholds rejected");
  c=measured_config(); c.local_support_radius_m=.5;
  invalid([&]{bev_geometry::validate_config(c);},"support radius smaller than patch rejected");
  c=measured_config(); c.self_x_min_m=2;
  invalid([&]{bev_geometry::validate_config(c);},"inverted body bounds rejected");
}
void sparse_floor_amid_roofs_and_prior_slopes() {
  std::vector<Point> scene;
  for (int x=15;x<=25;++x) {
    for (int y=-5;y<=5;++y) { scene.push_back({x*.4+.03,y*.4+.03,-1.85}); }
  }
  const auto floor_count=scene.size();
  // Hundreds of broad roof patches swamp the sparse 4 m by 4 m floor.
  // They are within the slope cone at long ranges, but not the height prior.
  for (int x=-21;x<=21;++x) {
    for (int y=-21;y<=21;++y) {
      for (int n=0;n<4;++n) { scene.push_back({x*.8+.02+n*.02,y*.8+.02+n*.01,-.3}); }
    }
  }
  std::mt19937 random(2401);
  std::normal_distribution<double> noise(0,.002);
  for (int scan=0;scan<12;++scan) {
    auto perturbed=scene;
    for (auto &p : perturbed) { p.x+=noise(random); p.y+=noise(random); p.z+=noise(random); }
    const auto result=bev_geometry::classify_and_grid(perturbed,measured_config());
    require(result.ground.valid,"small scan changes must not lose a sparse supported floor amid roofs");
    require(std::abs(result.ground.c+1.85)<.02,"roof stripes must not displace the real floor fit");
    require(std::count(result.labels.begin(),result.labels.begin()+floor_count,Label::Ground)>=
            static_cast<std::ptrdiff_t>(floor_count*.95),"sparse floor labels must remain stable");
  }
  for (const double intercept : {-1.52,-2.08}) {
    const auto result=bev_geometry::classify_and_grid(floor(.24,-.04,intercept),measured_config());
    require(result.ground.valid && std::abs(result.ground.c-intercept)<.01,
            "prior seed gate must preserve steep floors and height-tolerance offsets");
    require(std::abs(result.ground.a-.24)<.002 && std::abs(result.ground.b+.04)<.002,
            "height prior proposals must freely refine the actual slope");
  }
}
} // namespace

int main() {
  try {
    sloped_floor_box_and_dominant_wall();
    reject_unsupported_planes();
    occlusion_and_unknown_gaps();
    preserve_indices_and_explicit_masks();
    invalid_configuration();
    sparse_floor_amid_roofs_and_prior_slopes();
    std::cout << "ground geometry tests passed (6 groups)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
