cmake_minimum_required(VERSION 3.16)
project(bev_geometry VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(bev_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

add_library(bev_geometry src/ground.cpp)
add_library(bev_geometry::bev_geometry ALIAS bev_geometry)
target_compile_features(bev_geometry PUBLIC cxx_std_17)
target_include_directories(bev_geometry PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(bev_geometry PUBLIC bev_core::bev_core)
target_compile_options(bev_geometry PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ground_bev_node src/ground_bev_node.cpp)
target_compile_features(ground_bev_node PRIVATE cxx_std_17)
target_compile_options(ground_bev_node PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(ground_bev_node PRIVATE bev_geometry::bev_geometry)
ament_target_dependencies(ground_bev_node PUBLIC rclcpp sensor_msgs nav_msgs diagnostic_msgs)
install(TARGETS ground_bev_node DESTINATION lib/${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
  add_executable(ground_tests test/ground_tests.cpp)
  target_link_libraries(ground_tests PRIVATE bev_geometry::bev_geometry)
  target_compile_options(ground_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME ground_geometry_tests COMMAND ground_tests)
endif()

install(TARGETS bev_geometry EXPORT bev_geometryTargets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(bev_geometryTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(bev_core)
ament_package()
