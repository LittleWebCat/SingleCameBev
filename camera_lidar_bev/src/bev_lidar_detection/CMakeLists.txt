cmake_minimum_required(VERSION 3.18)
project(bev_lidar_detection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110)
endif()
option(BEV_LIDAR_STANDALONE "Build the native detector without ROS packaging" OFF)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
set(POINTPILLARS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/CUDA-PointPillars")
file(GLOB POINTPILLARS_SOURCES CONFIGURE_DEPENDS
  "${POINTPILLARS_ROOT}/src/pointpillar/*.cpp"
  "${POINTPILLARS_ROOT}/src/pointpillar/*.cu")
add_library(pointpillars_core SHARED
  ${POINTPILLARS_SOURCES}
  "${POINTPILLARS_ROOT}/src/common/tensorrt.cpp")
target_include_directories(pointpillars_core PRIVATE
  "${POINTPILLARS_ROOT}/src" "${POINTPILLARS_ROOT}/src/pointpillar"
  "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(pointpillars_core PRIVATE CUDA::cudart "${TENSORRT_LIBRARY}")
target_compile_options(pointpillars_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
add_library(bev_lidar_detector SHARED src/detector.cpp)
target_include_directories(bev_lidar_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(bev_lidar_detector PRIVATE
  "${POINTPILLARS_ROOT}/src" "${POINTPILLARS_ROOT}/src/pointpillar")
target_link_libraries(bev_lidar_detector PRIVATE pointpillars_core CUDA::cudart)
add_executable(pointpillars_benchmark src/pointpillars_benchmark.cpp)
target_link_libraries(pointpillars_benchmark PRIVATE bev_lidar_detector)

if(NOT BEV_LIDAR_STANDALONE)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(vision_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(diagnostic_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_msgs REQUIRED)
  add_executable(lidar_vehicle_detector src/lidar_vehicle_detector.cpp)
  target_compile_options(lidar_vehicle_detector PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(lidar_vehicle_detector PRIVATE bev_lidar_detector)
  ament_target_dependencies(lidar_vehicle_detector PUBLIC rclcpp sensor_msgs vision_msgs visualization_msgs diagnostic_msgs geometry_msgs tf2_msgs)
  install(TARGETS pointpillars_core bev_lidar_detector
    EXPORT bev_lidar_detector_targets LIBRARY DESTINATION lib)
  install(TARGETS pointpillars_benchmark lidar_vehicle_detector DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)
  ament_export_targets(bev_lidar_detector_targets HAS_LIBRARY_TARGET)
  ament_package()
endif()
