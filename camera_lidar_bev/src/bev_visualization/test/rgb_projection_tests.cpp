#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "bev_visualization/cloud_io.hpp"
#include "sensor_msgs/msg/point_field.hpp"

using namespace bev_visualization;
namespace {
void require(bool condition, const std::string &message) { if (!condition) throw std::runtime_error(message); }
void near(double actual, double expected, const std::string &message, double tolerance = 1e-6) {
  require(std::abs(actual-expected) < tolerance, message);
}
template<typename F> void rejects(F callback, const std::string &message) {
  bool rejected=false; try { callback(); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,message);
}
Camera small_camera() { return {8,8,{5,0,3,0,5,3,0,0,1},{0,0,0,0,0}}; }
ImageView image_view(const Camera &camera, const std::vector<std::uint8_t> &pixels, bool bgr=true) {
  return {pixels.data(),pixels.size(),static_cast<std::size_t>(camera.width)*3,camera.width,camera.height,bgr};
}
void test_composition_and_raw_colors() {
  auto camera=small_camera(); std::vector<std::uint8_t> pixels(camera.width*camera.height*3,0);
  const auto offset=(4*camera.width+3)*3; pixels[offset]=20;pixels[offset+1]=40;pixels[offset+2]=60;
  const Transform camera_from_sensor={-1,0,0,0,0,-1,0,0,0,0,1,-0.5,0,0,0,1};
  const Transform sensor_from_lidar={-1,0,0,0,0,-1,0,0,0,0,1,0.038195,0,0,0,1};
  const double nan=std::numeric_limits<double>::quiet_NaN();
  const std::vector<XYZ> points={{0,0,0},{.1,.2,2},{.2,.4,3.538195},{0,0,-1},{nan,0,1}};
  const auto result=colorize(points,camera,image_view(camera,pixels),camera_from_sensor,sensor_from_lidar);
  require(result.input_count==5 && result.points.size()==3 && result.colored_count==1,"valid/invalid returns or depth association incorrect");
  const auto &first=result.points[0];
  near(first.x,-.1,"os_sensor x did not apply observed 180-degree rotation");
  near(first.y,-.2,"os_sensor y did not apply observed 180-degree rotation");
  near(first.z,2.038195,"os_sensor origin offset missing");
  require(first.source_index==1 && first.camera_colored && first.rgb==0x003c2814,"source identity or native BGR pixel wrong");
  require(result.points[1].source_index==2 && result.points[1].rgb==0x00ffffff && !result.points[1].camera_colored,"occluded return was colored");
  require(result.points[2].source_index==3 && result.points[2].rgb==0x00ffffff,"behind-camera geometry lost or colored");
  const auto rgb=colorize(points,camera,image_view(camera,pixels,false),camera_from_sensor,sensor_from_lidar);
  require(rgb.points[0].rgb==0x0014283c,"native RGB encoding order wrong");
}
void test_zbuffer_and_ties() {
  const auto camera=small_camera(); std::vector<std::uint8_t> pixels(camera.width*camera.height*3,127);
  const auto identity=identity_transform();
  const auto result=colorize({{0,0,3},{0,0,1},{0,0,1}},camera,image_view(camera,pixels),identity,identity);
  require(result.colored_count==1 && !result.points[0].camera_colored && result.points[1].camera_colored && !result.points[2].camera_colored,
          "zbuffer must choose closest optical depth, then earliest source for ties");
}
void test_radial_branch_and_edges() {
  auto camera=small_camera();camera.d={-1,0,0,0,0};
  near(radial_domain_radius_squared(camera.d),1.0/3.0,"central radial derivative root wrong",1e-12);
  std::vector<std::uint8_t> pixels(camera.width*camera.height*3,200);
  const auto identity=identity_transform();
  const auto result=colorize({{1,0,1},{0,0,2}},camera,image_view(camera,pixels),identity,identity);
  require(result.radial_excluded_count==1 && !result.points[0].camera_colored && result.points[1].camera_colored,
          "folded outside ray stole a valid image pixel");
  camera.d={0,0,0,0,0};camera.k={1,0,0,0,1,0,0,0,1};
  const auto edge=colorize({{7.8,0,1},{8,0,1},{-.1,0,1}},camera,image_view(camera,pixels),identity,identity);
  require(edge.colored_count==1 && edge.points[0].camera_colored && !edge.points[1].camera_colored && !edge.points[2].camera_colored,
          "native floating bounds must precede nearest-pixel rounding/clamping");
  const std::array<double,5> imported={-.4430230293674048,.7644587241471862,.003543153617926078,-.00020622865673565534,-1.7672146488596931};
  near(std::sqrt(radial_domain_radius_squared(imported)),.677362,"imported radial central branch wrong",1e-6);
}
void test_layout_and_invalid_models() {
  auto camera=small_camera(); std::vector<std::uint8_t> pixels(camera.width*camera.height*3,0);
  auto view=image_view(camera,pixels);const auto identity=identity_transform();
  view.width=4; rejects([&]{colorize({{0,0,1}},camera,view,identity,identity);},"resized image accepted against native K");
  auto reflected=identity;reflected[0]=-1; rejects([&]{validate_transform(reflected);},"reflection accepted as rigid rotation");
  camera.k[1]=1;rejects([&]{validate_camera(camera);},"unsupported skew accepted");
}
template<typename T> void write_be(std::uint8_t *destination, T value) {
  std::array<std::uint8_t,sizeof(T)> bytes{};std::memcpy(bytes.data(),&value,sizeof(T));
  const std::uint16_t one=1;const bool host_little=*reinterpret_cast<const std::uint8_t*>(&one)==1;
  for(std::size_t i=0;i<sizeof(T);++i) destination[i]=bytes[host_little ? sizeof(T)-1-i : i];
}
void test_padded_big_endian_cloud_and_rgb_fields() {
  using Field=sensor_msgs::msg::PointField;
  sensor_msgs::msg::PointCloud2 cloud;cloud.width=1;cloud.height=2;cloud.point_step=24;cloud.row_step=32;cloud.is_bigendian=true;cloud.data.resize(64);
  for(int i=0;i<3;++i) {Field field;field.name=i==0?"x":i==1?"y":"z";field.offset=i*8;field.datatype=Field::FLOAT64;field.count=1;cloud.fields.push_back(field);}
  write_be(cloud.data.data(),1.25);write_be(cloud.data.data()+8,-2.5);write_be(cloud.data.data()+16,3.0);
  write_be(cloud.data.data()+32,4.5);write_be(cloud.data.data()+40,5.0);write_be(cloud.data.data()+48,6.0);
  const auto decoded=decode_xyz(cloud);require(decoded.size()==2,"row padding changed cloud point count");
  near(decoded[0].x,1.25,"big-endian FLOAT64 x wrong");near(decoded[0].y,-2.5,"big-endian FLOAT64 y wrong");near(decoded[1].z,6,"row padding ignored");
  auto bad=cloud;bad.fields[0].offset=23;rejects([&]{decode_xyz(bad);},"out-of-record XYZ field accepted");
  bad=cloud;bad.data.pop_back();rejects([&]{decode_xyz(bad);},"short cloud payload accepted");
  std_msgs::msg::Header header;header.frame_id="os_sensor";
  const auto packed=encode_rgb({{1,2,3,0x00ab1234,7,true},{4,5,6,0x00ffffff,9,false}},header);
  require(packed.width==2 && packed.height==1 && packed.point_step==24 && packed.fields.size()==6,"RGB field layout incorrect");
  require(!packed.is_bigendian && packed.fields[3].name=="rgb" && packed.fields[3].datatype==Field::FLOAT32 && packed.fields[4].datatype==Field::UINT32 && packed.fields[5].datatype==Field::UINT8,"RGB/source/mask datatypes incorrect");
  require(packed.data[12]==0x34 && packed.data[13]==0x12 && packed.data[14]==0xab && packed.data[15]==0 && packed.data[16]==7 && packed.data[20]==1,
          "packed RGB bit order or source/mask bytes wrong");
  require(packed.data[24+20]==0 && packed.data[21]==0 && packed.data[22]==0 && packed.data[23]==0,"white fallback/mask or padding incorrect");
  const auto empty=encode_rgb({},header);require(empty.width==0 && empty.data.empty() && empty.fields.size()==6 && empty.header.frame_id=="os_sensor","clear-output message loses RGB fields/frame");
}
}  // namespace
int main() {
  try {
    test_composition_and_raw_colors();test_zbuffer_and_ties();test_radial_branch_and_edges();
    test_layout_and_invalid_models();test_padded_big_endian_cloud_and_rgb_fields();
    std::cout<<"RGB projection tests passed: composition, native colors, occlusion, radial fold, bounds, endian/layout and packed fields\n";
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
  return 0;
}
