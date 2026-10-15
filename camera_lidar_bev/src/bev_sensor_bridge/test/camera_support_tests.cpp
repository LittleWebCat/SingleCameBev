#include "bev_sensor_bridge/camera_support.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace
{

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<typename Exception, typename Function>
void require_throws(Function function, const std::string & message)
{
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

class TemporaryCalibration
{
public:
  TemporaryCalibration()
  {
    std::array<char, 40> name{};
    const std::string pattern = "/tmp/bev-camera-support-XXXXXX";
    std::copy(pattern.begin(), pattern.end(), name.begin());
    const int descriptor = mkstemp(name.data());
    if (descriptor < 0) {
      throw std::runtime_error("cannot create temporary calibration fixture");
    }
    close(descriptor);
    path_ = name.data();
  }

  ~TemporaryCalibration()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void write(const std::string & yaml) const
  {
    std::ofstream file(path_);
    file << yaml;
    file.close();
    require(static_cast<bool>(file), "cannot write calibration fixture");
  }

  const std::string & path() const {return path_;}

private:
  std::string path_;
};

const std::string valid_yaml = R"(image_width: 640
image_height: 480
camera_name: front_camera
camera_matrix:
  rows: 3
  cols: 3
  data: [500.0, 0.0, 320.0, 0.0, 501.0, 240.0, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.001, -0.002, 0.03]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [502.0, 0.0, 321.0, 0.0, 0.0, 503.0, 241.0, 0.0, 0.0, 0.0, 1.0, 0.0]
)";

std::string replace(const std::string & original, const std::string & from, const std::string & to)
{
  std::string modified = original;
  const auto position = modified.find(from);
  require(position != std::string::npos, "fixture replacement must find its source");
  modified.replace(position, from.size(), to);
  return modified;
}

std_msgs::msg::Header acquisition_header()
{
  std_msgs::msg::Header header;
  header.stamp.sec = 17;
  header.stamp.nanosec = 123456789;
  header.frame_id = "front_camera_optical_frame";
  return header;
}

void require_uncalibrated(const sensor_msgs::msg::CameraInfo & info)
{
  require(info.k[0] == 0.0, "uncalibrated CameraInfo must have K[0] = 0");
  require(info.distortion_model.empty(), "uncalibrated distortion model must be unset");
  require(info.d.empty(), "uncalibrated distortion coefficients must be unset");
  for (const auto value : info.k) {require(value == 0.0, "uncalibrated K must be zero");}
  for (const auto value : info.r) {require(value == 0.0, "uncalibrated R must be zero");}
  for (const auto value : info.p) {require(value == 0.0, "uncalibrated P must be zero");}
}

void calibration_absent_and_matching_header()
{
  const auto calibration = bev_sensor_bridge::load_camera_calibration("");
  require(!calibration.calibrated, "empty path must select uncalibrated mode");
  const auto header = acquisition_header();
  const auto result = bev_sensor_bridge::camera_info_for_frame(calibration, 1280, 720, header);
  require_uncalibrated(result.message);
  require(!result.calibrated && result.error.empty(), "intentional uncalibrated mode is not an error");
  require(result.message.width == 1280 && result.message.height == 720, "actual capture dimensions");
  require(result.message.header == header, "CameraInfo must preserve acquisition header");
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::camera_info_for_frame(calibration, 0, 720, header);
    }, "zero capture width must throw");
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::camera_info_for_frame(calibration, 1280, 0, header);
    }, "zero capture height must throw");
}

void calibrated_yaml_and_resolution_mismatch()
{
  TemporaryCalibration fixture;
  fixture.write(valid_yaml);
  const auto calibration = bev_sensor_bridge::load_camera_calibration(fixture.path());
  require(calibration.calibrated, "valid YAML must mark calibration valid");
  require(calibration.message.width == 640 && calibration.message.height == 480, "calibration dimensions");
  require(calibration.message.k[0] == 500.0 && calibration.message.k[4] == 501.0, "load focal lengths");
  require(calibration.message.d.size() == 5 && calibration.message.d[1] == -0.2, "load distortion");
  require(calibration.message.p[0] == 502.0 && calibration.message.p[2] == 321.0, "load rectified intrinsics");
  const auto header = acquisition_header();
  const auto matched = bev_sensor_bridge::camera_info_for_frame(calibration, 640, 480, header);
  require(matched.calibrated && matched.error.empty(), "matching capture must preserve calibration");
  require(matched.message.header == header && matched.message.k == calibration.message.k,
    "matching calibration must retain matrix and frame header");

  for (const auto & size : {std::array<std::uint32_t, 2>{1280, 480}, {640, 720}}) {
    const auto mismatched = bev_sensor_bridge::camera_info_for_frame(calibration, size[0], size[1], header);
    require(!mismatched.calibrated && !mismatched.error.empty(), "resolution mismatch must report error");
    require_uncalibrated(mismatched.message);
    require(mismatched.message.header == header, "mismatch must retain acquisition header");
    require(mismatched.message.width == size[0] && mismatched.message.height == size[1],
      "mismatch must report actual dimensions without scaling");
  }

  auto rational = replace(valid_yaml, "plumb_bob", "rational_polynomial");
  rational = replace(rational, "cols: 5", "cols: 8");
  rational = replace(rational, "[0.1, -0.2, 0.001, -0.002, 0.03]", "[0.1, -0.2, 0.001, -0.002, 0.03, 0.0, 0.0, 0.0]");
  fixture.write(rational);
  require(bev_sensor_bridge::load_camera_calibration(fixture.path()).message.d.size() == 8,
    "rational model must load eight coefficients");
  auto equidistant = replace(valid_yaml, "plumb_bob", "equidistant");
  equidistant = replace(equidistant, "cols: 5", "cols: 4");
  equidistant = replace(equidistant, "[0.1, -0.2, 0.001, -0.002, 0.03]", "[0.1, -0.2, 0.001, -0.002]");
  fixture.write(equidistant);
  require(bev_sensor_bridge::load_camera_calibration(fixture.path()).message.d.size() == 4,
    "equidistant model must load four coefficients");
}

void invalid_calibration_is_explicit()
{
  TemporaryCalibration fixture;
  const auto reject = [&](const std::string & yaml) {
      fixture.write(yaml);
      require_throws<std::runtime_error>([&] {bev_sensor_bridge::load_camera_calibration(fixture.path());},
        "invalid supplied calibration must throw rather than fabricate intrinsics");
    };
  for (const std::string dimension : {"0", "-1", "640.5", ".nan", ".inf", "4294967296", "word"}) {
    reject(replace(valid_yaml, "image_width: 640", "image_width: " + dimension));
    reject(replace(valid_yaml, "image_height: 480", "image_height: " + dimension));
  }
  reject("[]");
  reject("image_width: [broken");
  reject(replace(valid_yaml, "camera_matrix:", "omitted_camera_matrix:"));
  reject(replace(valid_yaml, "rectification_matrix:", "omitted_rectification_matrix:"));
  reject(replace(valid_yaml, "projection_matrix:", "omitted_projection_matrix:"));
  reject(replace(valid_yaml, "distortion_model: plumb_bob", "distortion_model: unsupported"));
  reject(replace(valid_yaml, "distortion_model: plumb_bob", "distortion_model: equidistant"));
  reject(replace(valid_yaml, "distortion_model: plumb_bob", "distortion_model: rational_polynomial"));
  reject(replace(valid_yaml, "rows: 3", "rows: 1"));
  reject(replace(valid_yaml, "cols: 3", "cols: 9"));
  reject(replace(valid_yaml, "cols: 5", "cols: 4"));
  reject(replace(valid_yaml, "cols: 4", "cols: 3"));
  reject(replace(valid_yaml, "[0.1, -0.2, 0.001, -0.002, 0.03]", "[0.1, -0.2, 0.001, -0.002]"));
  for (const std::string value : {"0.0", "-1.0", ".nan", ".inf", "word"}) {
    reject(replace(valid_yaml, "500.0", value));
    reject(replace(valid_yaml, "503.0", value));
  }
  reject(replace(valid_yaml, "0.03]", ".nan]"));
  reject(replace(valid_yaml, "1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"));
  reject(replace(valid_yaml, "1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0", "-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0"));
  reject(replace(valid_yaml, "0.0, 0.0, 1.0, 0.0]", "0.0, 0.0, 0.0, 0.0]"));
  const auto missing_path = fixture.path() + ".missing";
  require_throws<std::runtime_error>([&] {bev_sensor_bridge::load_camera_calibration(missing_path);},
    "supplied missing path must throw");
}

void bgr_layout_and_noncontiguous_roi()
{
  cv::Mat storage(4, 5, CV_8UC3);
  for (int row = 0; row < storage.rows; ++row) {
    for (int column = 0; column < storage.cols; ++column) {
      storage.at<cv::Vec3b>(row, column) = cv::Vec3b(
        static_cast<unsigned char>(row * 10 + column),
        static_cast<unsigned char>(80 + row), static_cast<unsigned char>(160 + column));
    }
  }
  const cv::Mat roi = storage(cv::Rect(1, 1, 3, 2));
  require(!roi.isContinuous(), "test must exercise an OpenCV ROI with parent row padding");
  const auto header = acquisition_header();
  const auto message = bev_sensor_bridge::bgr_image_message(roi, header);
  require(message.header == header, "image must retain acquisition header");
  require(message.width == 3 && message.height == 2 && message.step == 9, "BGR8 tight row layout");
  require(message.encoding == "bgr8" && message.is_bigendian == 0, "BGR8 encoding metadata");
  require(message.data.size() == 18, "image data must be step * height");
  for (int row = 0; row < roi.rows; ++row) {
    for (int column = 0; column < roi.cols; ++column) {
      const auto pixel = roi.at<cv::Vec3b>(row, column);
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const auto index = static_cast<std::size_t>(row) * message.step +
          static_cast<std::size_t>(column) * 3 + channel;
        require(message.data[index] == pixel[static_cast<int>(channel)], "ROI copy must preserve BGR ordering");
      }
    }
  }
  const auto continuous = bev_sensor_bridge::bgr_image_message(roi.clone(), header);
  require(continuous.data == message.data, "contiguous and strided copies must agree");
  const auto info = bev_sensor_bridge::camera_info_for_frame(
    bev_sensor_bridge::load_camera_calibration(""), message.width, message.height, header);
  require(info.message.header == message.header, "image and CameraInfo must share header exactly");
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::bgr_image_message(cv::Mat{}, header);
    }, "empty image must throw");
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::bgr_image_message(cv::Mat(2, 2, CV_8UC1), header);
    }, "grayscale frame must throw rather than mislabel encoding");
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::bgr_image_message(cv::Mat(2, 2, CV_16UC3), header);
    }, "non-byte BGR frame must throw");
  const int sizes[] = {2, 2, 2};
  require_throws<std::invalid_argument>([&] {
      bev_sensor_bridge::bgr_image_message(cv::Mat(3, sizes, CV_8UC3), header);
    }, "three-dimensional frame must throw");
}

}  // namespace

int main()
{
  try {
    calibration_absent_and_matching_header();
    calibrated_yaml_and_resolution_mismatch();
    invalid_calibration_is_explicit();
    bgr_layout_and_noncontiguous_roi();
  } catch (const std::exception & error) {
    std::cerr << "camera support test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "camera calibration, resolution, header, and BGR layout tests passed\n";
  return EXIT_SUCCESS;
}
