#include "bev_visualization/cloud_io.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "sensor_msgs/msg/point_field.hpp"

namespace bev_visualization {
namespace {
bool host_big_endian() { const std::uint16_t word = 1; return *reinterpret_cast<const std::uint8_t*>(&word) == 0; }
template<typename T> T read_value(const std::uint8_t *bytes, bool big_endian) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::copy(bytes, bytes+sizeof(T), storage.begin());
  if (big_endian != host_big_endian()) std::reverse(storage.begin(), storage.end());
  T value; std::memcpy(&value, storage.data(), sizeof(T)); return value;
}
template<typename T> void write_little(std::uint8_t *destination, T value) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::memcpy(storage.data(), &value, sizeof(T));
  if (host_big_endian()) std::reverse(storage.begin(), storage.end());
  std::copy(storage.begin(), storage.end(), destination);
}
}  // namespace

std::vector<XYZ> decode_xyz(const sensor_msgs::msg::PointCloud2 &cloud) {
  using Field = sensor_msgs::msg::PointField;
  const auto count = static_cast<std::uint64_t>(cloud.width)*cloud.height;
  if (!cloud.width || !cloud.height || !cloud.point_step || count > std::numeric_limits<std::uint32_t>::max() ||
      static_cast<std::uint64_t>(cloud.width)*cloud.point_step > cloud.row_step ||
      static_cast<std::uint64_t>(cloud.row_step)*cloud.height != cloud.data.size())
    throw std::invalid_argument("PointCloud2 payload has invalid row/point layout");
  std::array<const Field*, 3> fields{};
  for (const auto &field : cloud.fields) {
    const int axis = field.name == "x" ? 0 : field.name == "y" ? 1 : field.name == "z" ? 2 : -1;
    if (axis < 0) continue;
    if (fields[axis] || field.count != 1 || (field.datatype != Field::FLOAT32 && field.datatype != Field::FLOAT64))
      throw std::invalid_argument("XYZ fields must be unique scalar FLOAT32/FLOAT64");
    const auto bytes = field.datatype == Field::FLOAT32 ? 4U : 8U;
    if (static_cast<std::uint64_t>(field.offset)+bytes > cloud.point_step)
      throw std::invalid_argument("XYZ field exceeds point_step");
    fields[axis] = &field;
  }
  if (!fields[0] || !fields[1] || !fields[2]) throw std::invalid_argument("Cloud is missing XYZ");
  std::vector<XYZ> result; result.reserve(static_cast<std::size_t>(count));
  for (std::uint32_t row = 0; row < cloud.height; ++row) for (std::uint32_t column = 0; column < cloud.width; ++column) {
    const auto *point = cloud.data.data()+static_cast<std::size_t>(row)*cloud.row_step+static_cast<std::size_t>(column)*cloud.point_step;
    std::array<double, 3> xyz{};
    for (int axis = 0; axis < 3; ++axis) {
      const auto *field = fields[axis];
      xyz[axis] = field->datatype == Field::FLOAT32 ?
        static_cast<double>(read_value<float>(point+field->offset, cloud.is_bigendian)) :
        read_value<double>(point+field->offset, cloud.is_bigendian);
    }
    result.push_back({xyz[0], xyz[1], xyz[2]});
  }
  return result;
}

sensor_msgs::msg::PointCloud2 encode_rgb(const std::vector<ColoredPoint> &points, const std_msgs::msg::Header &header) {
  using Field = sensor_msgs::msg::PointField;
  if (points.size() > std::numeric_limits<std::uint32_t>::max()/24)
    throw std::invalid_argument("RGB cloud exceeds UINT32 row_step");
  sensor_msgs::msg::PointCloud2 message;
  message.header = header; message.height = 1;
  message.width = static_cast<std::uint32_t>(points.size());
  message.is_bigendian = false; message.point_step = 24;
  message.row_step = message.width*message.point_step; message.is_dense = true;
  for (const auto &entry : std::array<std::pair<const char*, std::uint8_t>, 6>{{
         {"x",Field::FLOAT32},{"y",Field::FLOAT32},{"z",Field::FLOAT32},
         {"rgb",Field::FLOAT32},{"source_index",Field::UINT32},{"camera_colored",Field::UINT8}}}) {
    Field field; field.name = entry.first; field.datatype = entry.second; field.count = 1;
    field.offset = static_cast<std::uint32_t>(message.fields.size())*4;
    message.fields.push_back(field);
  }
  message.data.resize(message.row_step, 0);
  for (std::size_t index = 0; index < points.size(); ++index) {
    const auto &point = points[index]; auto *bytes = message.data.data()+index*24;
    write_little(bytes, point.x); write_little(bytes+4, point.y); write_little(bytes+8, point.z);
    write_little(bytes+12, point.rgb); write_little(bytes+16, point.source_index);
    bytes[20] = point.camera_colored ? 1 : 0;
  }
  return message;
}
}  // namespace bev_visualization
