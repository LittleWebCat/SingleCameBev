cmake_minimum_required(VERSION 3.16)
project(bev_bringup)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY ../../config/ DESTINATION share/${PROJECT_NAME}/config)
install(FILES ../../tools/combined_bev_live.py ../../tools/combined_bev_display.py
  ../../tools/saved_map_source.py
  ../../tools/live_map_source.py
  ../../tools/path_tracking_overlay.py
  ../../tools/tracking_history.py
  ../../tools/align_local_map_gps.py
  ../../tools/localization_map_view.py
  ../../tools/lidar_map_localizer.py
  ../../tools/road_lane_detector.py
  ../../tools/septentrio_gps.py
  DESTINATION share/${PROJECT_NAME}/tools)
ament_package()
