#include "bev_visualization/camera_box_association.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bev_visualization;
namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message, double tolerance=1e-8) {
  require(std::isfinite(actual) && std::abs(actual-expected)<tolerance, message);
}
template<typename F> void rejects(F function, const std::string &message) {
  bool rejected=false;
  try { function(); } catch (const std::invalid_argument &) { rejected=true; }
  require(rejected, message);
}
Camera camera() { return {640,480,{400,0,320,0,400,240,0,0,1},{0,0,0,0,0}}; }
OrientedBox box() { return {{0,0,10},{4,2,2},{0,0,0,1}}; }
CameraDetection detection(const ProjectedBox &p, const std::string &name="car", double confidence=.9) {
  return {name,confidence,p.x_min,p.y_min,p.x_max,p.y_max};
}
bool unchanged(const OrientedBox &a, const OrientedBox &b) {
  return a.center.x==b.center.x && a.center.y==b.center.y && a.center.z==b.center.z &&
         a.size.x==b.size.x && a.size.y==b.size.y && a.size.z==b.size.z &&
         a.orientation_xyzw==b.orientation_xyzw && a.class_name==b.class_name;
}

void test_pinhole_corners_and_rotation() {
  auto b=box(); const auto c=camera(); const auto t=identity_transform();
  auto p=project_oriented_box(b,c,t);
  require(p.visible,"Front-facing complete cuboid should project");
  near(p.x_min,320-400*2.0/9,"Near front corner must set the left extent");
  near(p.x_max,320+400*2.0/9,"Near front corner must set the right extent");
  near(p.y_min,240-400*1.0/9,"Full height projection lost the near face");
  near(p.y_max,240+400*1.0/9,"Full height projection lost the near face");
  near(p.center_u,320,"Optical box center is wrong"); near(p.center_v,240,"Optical box center is wrong");
  b.orientation_xyzw={0,0,std::sqrt(.5),std::sqrt(.5)};
  p=project_oriented_box(b,c,t);
  near(p.x_min,320-400*1.0/9,"90-degree quaternion did not rotate the cuboid width");
  near(p.y_min,240-400*2.0/9,"90-degree quaternion did not rotate the cuboid length");
  // A nontrivial sensor -> camera transform checks the os_sensor convention:
  // sensor x forward, y left, z up; camera x right, y down, z forward.
  const Transform optical={0,-1,0,.2,0,0,-1,-.1,1,0,0,-.5,0,0,0,1};
  b={{10,0,0},{4,2,2},{0,0,0,1}};
  p=project_oriented_box(b,c,optical);
  require(p.visible,"Proper camera-from-sensor transform rejected");
  near(p.center_u,320+400*.2/9.5,"Camera translation/convention applied incorrectly");
  near(p.center_v,240-400*.1/9.5,"Camera optical y convention applied incorrectly");
  near(p.x_min,320-400*.8/7.5,"Sensor-forward depth must use sensor x");
  near(p.x_max,320+400*1.2/7.5,"LiDAR-to-camera transform was inverted or applied twice");
}

void test_native_distortion_fixture() {
  auto c=camera(); c.d={.1,.02,.005,-.003,.001};
  auto b=box(); b.center={1,.5,10};
  const auto p=project_oriented_box(b,c,identity_transform());
  require(p.visible,"Regular native plumb_bob projection rejected");
  // Independent numeric fixture from the eight native pinhole corner rays,
  // with Brown-Conrady radial/tangential coefficients (no detector resize).
  near(p.x_min,275.22394334329783,"Native distorted left bound differs from fixture");
  near(p.y_min,217.77177336072532,"Native distorted top bound differs from fixture");
  near(p.x_max,455.02587162780065,"Native distorted right bound differs from fixture");
  near(p.y_max,307.87404692501144,"Native distorted bottom bound differs from fixture");
  near(p.center_u,360.031125078125,"Tangential distortion or native principal point missing");
  near(p.center_v,260.04806253906247,"Native center distortion differs from fixture");
}

void test_projection_visibility_and_invalid_geometry() {
  const auto c=camera(); const auto t=identity_transform(); auto b=box();
  b.center.z=1;
  require(!project_oriented_box(b,c,t).visible,"Near-plane-crossing cuboid must be invisible");
  b=box();b.center.z=-10;
  require(!project_oriented_box(b,c,t).visible,"Behind-camera cuboid must be invisible");
  b=box();b.center.x=20;
  require(!project_oriented_box(b,c,t).visible,"Out-of-FOV center must not associate");
  b=box();b.center.x=7;b.size.x=8;
  const auto clipped=project_oriented_box(b,c,t);
  require(clipped.visible,"Partially in-frame cuboid with visible center was rejected");
  near(clipped.x_max,640,"Projected ROI must clip to native image bounds");
  near(clipped.x_min,320+400*3.0/11,"Clipped box lost its visible corner geometry");
  auto folded=c;folded.d={-1,0,0,0,0};b=box();b.center.z=3;b.size={3,1,1};
  require(!project_oriented_box(b,folded,t).visible,"Radial-fold corner must not make a valid image rectangle");
  b=box();b.orientation_xyzw={0,0,0,0};
  require(!project_oriented_box(b,c,t).visible,"Zero quaternion accepted");
  b.orientation_xyzw={0,0,0,2};
  require(!project_oriented_box(b,c,t).visible,"Non-unit quaternion accepted");
  b=box();b.size.x=0;
  require(!project_oriented_box(b,c,t).visible,"Zero-size cuboid accepted");
  b=box();b.center.y=std::numeric_limits<double>::quiet_NaN();
  require(!project_oriented_box(b,c,t).visible,"Nonfinite cuboid accepted");
}

void test_unique_matches_and_geometry_preserved() {
  const auto c=camera(); const auto t=identity_transform();
  auto left=box(), right=box();left.center.x=-3;right.center.x=3;left.size.x=right.size.x=2;
  const auto lp=project_oriented_box(left,c,t),rp=project_oriented_box(right,c,t);
  const std::vector<OrientedBox> boxes{left,right};const auto before=boxes;
  const std::vector<CameraDetection> camera_boxes{detection(rp,"truck",.81),detection(lp,"car",.73)};
  const auto result=associate_camera_boxes(boxes,camera_boxes,c,t);
  require(result.size()==2 && result[0].camera_index==1 && result[1].camera_index==0,
          "Unique matches must address original camera indices");
  near(result[0].iou,1,"Exact ROI IoU is wrong");near(result[1].iou,1,"Exact ROI IoU is wrong");
  require(unchanged(boxes[0],before[0]) && unchanged(boxes[1],before[1]),
          "Association altered predicted center, dimensions or quaternion");
  require(camera_boxes[0].class_name=="truck" && camera_boxes[0].confidence==.81,
          "Camera class/score must remain separate evidence");
  const auto absent=associate_camera_boxes(boxes,{},c,t);
  require(absent.size()==2 && absent[0].camera_index==-1 && absent[1].camera_index==-1 && absent[0].projection.visible,
          "Missing camera boxes must retain LiDAR projections as unmatched");
}

void test_supported_classes_confidence_and_bad_camera_boxes() {
  const auto c=camera();const auto t=identity_transform();const auto b=box();
  const auto p=project_oriented_box(b,c,t);
  for (const auto *name : {"car","motorcycle","bus","truck"}) {
    const auto result=associate_camera_boxes({b},{detection(p,name)},c,t);
    require(result[0].camera_index==0,"Supported vehicle camera class was rejected");
  }
  auto low=detection(p,"car",.34), over=detection(p,"car",1.1);
  auto malformed=detection(p);malformed.x_min=-1;
  auto invalid=detection(p);invalid.y_max=std::numeric_limits<double>::quiet_NaN();
  auto reversed=detection(p);reversed.x_max=reversed.x_min;
  const auto result=associate_camera_boxes({b},
      {detection(p,"person"),detection(p,"bicycle"),low,over,malformed,invalid,reversed},c,t);
  require(result[0].camera_index==-1,"Unsupported, low-confidence or malformed camera evidence was matched");
  const auto indexed=associate_camera_boxes({b},{detection(p,"person"),low,detection(p)},c,t);
  require(indexed[0].camera_index==2,"Filtering camera evidence changed original camera index");
}

void test_overlap_and_projected_center_gates() {
  const auto c=camera();const auto t=identity_transform();const auto b=box();
  const auto p=project_oriented_box(b,c,t);
  CameraDetection enormous{"car",.9,0,0,640,480};
  require(associate_camera_boxes({b},{enormous},c,t)[0].camera_index==-1,
          "Low IoU camera rectangle must not associate solely by center containment");
  auto shifted=detection(p);shifted.x_min=p.center_u+1;
  require(associate_camera_boxes({b},{shifted},c,t)[0].camera_index==-1,
          "Large overlap without projected-center containment must be rejected");
}

void test_person_matches_and_separate_class_families() {
  const auto c=camera();const auto t=identity_transform();
  auto person=box();person.class_name="person";person.size={.7,.6,1.8};
  const auto before=person;const auto p=project_oriented_box(person,c,t);
  const auto camera_person=detection(p,"person",.83);
  const auto matched=associate_camera_boxes({person},{camera_person},c,t);
  require(matched[0].camera_index==0,"Genuine LiDAR person/camera person must associate");
  near(matched[0].iou,1,"Exact person ROI IoU is wrong");
  require(unchanged(person,before) && camera_person.class_name=="person" && camera_person.confidence==.83,
          "Person association changed predicted geometry/class or camera evidence");
  for (const auto *vehicle_name : {"car","motorcycle","bus","truck"}) {
    require(associate_camera_boxes({person},{detection(p,vehicle_name)},c,t)[0].camera_index==-1,
            "LiDAR person must not associate with vehicle camera evidence");
  }
  auto car=person;car.class_name="car";
  require(associate_camera_boxes({car},{camera_person},c,t)[0].camera_index==-1,
          "LiDAR car must not associate with camera person evidence");
  // Coincident boxes do not imply an overlap ambiguity across incompatible
  // class families; only eligible candidates contribute to each best match.
  const auto mixed=associate_camera_boxes({person,car},{detection(p,"car"),camera_person},c,t);
  require(mixed[0].camera_index==1 && mixed[1].camera_index==0,
          "Coincident car/person projections created cross-class ambiguity");
  for (const auto *unsupported : {"cyclist","bicycle","truck","unknown",""}) {
    auto other=person;other.class_name=unsupported;
    const auto result=associate_camera_boxes({other},{camera_person,detection(p,"car")},c,t);
    require(result[0].camera_index==-1 && result[0].projection.visible,
            "Unsupported LiDAR class must retain its projection while staying unmatched");
  }
}

void test_row_column_ambiguity_and_mutual_best() {
  const auto c=camera();const auto t=identity_transform();auto b=box();
  const auto p=project_oriented_box(b,c,t);const auto exact=detection(p);
  auto almost=exact;almost.x_min-=1;almost.x_max+=1;
  require(associate_camera_boxes({b},{exact,almost},c,t)[0].camera_index==-1,
          "Two nearly equal camera candidates must not receive an arbitrary row match");
  auto nearby=b;nearby.center.x=.01;
  const auto columns=associate_camera_boxes({b,nearby},{exact},c,t);
  require(columns[0].camera_index==-1 && columns[1].camera_index==-1,
          "One camera candidate with nearly equal LiDAR overlaps must stay ambiguous");
  auto worse=exact;worse.x_min-=30;worse.x_max+=30;
  const auto distinct=associate_camera_boxes({b},{exact,worse},c,t);
  require(distinct[0].camera_index==0,"Clearly strongest unique camera candidate should survive");
  nearby=b;nearby.center.x=.5;
  const auto mutual=associate_camera_boxes({b,nearby},{exact},c,t);
  require(mutual[0].camera_index==0 && mutual[1].camera_index==-1,
          "One camera box cannot be reused by a non-mutual-best LiDAR candidate");
  AssociationConfig no_margin;no_margin.ambiguity_margin=0;
  require(associate_camera_boxes({b},{exact,exact},c,t,no_margin)[0].camera_index==-1,
          "Exact ties remain ambiguous even with zero configured margin");
}

void test_invalid_global_calibration_and_configuration() {
  auto c=camera();const auto t=identity_transform();auto config=AssociationConfig{};
  c.k[1]=1;
  rejects([&]{associate_camera_boxes({box()},{},c,t);},"Unsupported skew accepted");
  c=camera();auto reflection=t;reflection[0]=-1;
  rejects([&]{associate_camera_boxes({box()},{},c,reflection);},"Improper calibration reflection accepted");
  config.min_iou=0;
  rejects([&]{associate_camera_boxes({box()},{},c,t,config);},"Zero IoU threshold accepted");
  config={};config.ambiguity_margin=-.01;
  rejects([&]{validate_association_config(config);},"Negative ambiguity margin accepted");
  config={};config.min_confidence=std::numeric_limits<double>::infinity();
  rejects([&]{validate_association_config(config);},"Nonfinite confidence threshold accepted");
  rejects([&]{project_oriented_box(box(),c,t,0);},"Nonpositive optical near plane accepted");
}
}  // namespace

int main() {
  try {
    test_pinhole_corners_and_rotation();
    test_native_distortion_fixture();
    test_projection_visibility_and_invalid_geometry();
    test_unique_matches_and_geometry_preserved();
    test_supported_classes_confidence_and_bad_camera_boxes();
    test_overlap_and_projected_center_gates();
    test_person_matches_and_separate_class_families();
    test_row_column_ambiguity_and_mutual_best();
    test_invalid_global_calibration_and_configuration();
    std::cout << "Camera box association tests passed: 9 groups, native geometry/distortion, visibility, immutable boxes, vehicle/person class gates, unique overlap and ambiguity\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
