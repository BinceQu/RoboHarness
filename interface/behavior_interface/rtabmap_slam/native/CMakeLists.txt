cmake_minimum_required(VERSION 3.14)
project(behavior_rtabmap_worker LANGUAGES C CXX)

find_package(RTABMap REQUIRED COMPONENTS core)
find_package(SQLite3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Ceres REQUIRED)

add_executable(behavior_rtabmap_worker worker.cpp)
target_compile_features(behavior_rtabmap_worker PRIVATE cxx_std_17)
target_compile_options(behavior_rtabmap_worker PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(
  behavior_rtabmap_worker
  PRIVATE
    rtabmap::core
    SQLite::SQLite3
    ZLIB::ZLIB
    Ceres::ceres
)
install(TARGETS behavior_rtabmap_worker RUNTIME DESTINATION bin)
