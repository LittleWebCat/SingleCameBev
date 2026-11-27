#pragma once

#include "bev_visualization/rgb_projection.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace bev_visualization {
// Honors row padding and byte order; accepts scalar FLOAT32/FLOAT64 XYZ.
std::vector<XYZ> decode_xyz(const sensor_msgs::msg::PointCloud2 &cloud);
// 24 bytes/point: x,y,z FLOAT32; rgb FLOAT32 packed 0x00RRGGBB;
// source_index UINT32; camera_colored UINT8; three zero padding bytes.
sensor_msgs::msg::PointCloud2 encode_rgb(const std::vector<ColoredPoint> &points,
                                      const std_msgs::msg::Header &header);
}  // namespace bev_visualization
