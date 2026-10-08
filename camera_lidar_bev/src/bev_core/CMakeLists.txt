cmake_minimum_required(VERSION 3.16)
project(bev_core VERSION 0.1.0 LANGUAGES CXX)

option(BEV_CORE_USE_AMENT "Use ament_cmake when available" ON)
if(BEV_CORE_USE_AMENT)
  find_package(ament_cmake QUIET)
endif()

add_library(bev_core src/grid.cpp)
add_library(bev_core::bev_core ALIAS bev_core)
target_compile_features(bev_core PUBLIC cxx_std_17)
target_include_directories(bev_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(bev_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(bev_core_grid_tests test/grid_tests.cpp)
  target_link_libraries(bev_core_grid_tests PRIVATE bev_core::bev_core)
  add_test(NAME bev_core_grid_tests COMMAND bev_core_grid_tests)
endif()

install(TARGETS bev_core EXPORT bev_coreTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BEV_CORE_USE_AMENT AND ament_cmake_FOUND)
  ament_export_targets(bev_coreTargets HAS_LIBRARY_TARGET)
  ament_package()
else()
  include(CMakePackageConfigHelpers)
  configure_package_config_file(cmake/bev_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bev_coreConfig.cmake
    INSTALL_DESTINATION lib/cmake/bev_core)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/bev_coreConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(EXPORT bev_coreTargets FILE bev_coreTargets.cmake
    NAMESPACE bev_core:: DESTINATION lib/cmake/bev_core)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/bev_coreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/bev_coreConfigVersion.cmake
    DESTINATION lib/cmake/bev_core)
endif()
