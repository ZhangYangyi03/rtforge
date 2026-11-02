cmake_minimum_required(VERSION 3.16)
project(rtforge_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -O2)

add_library(rtforge_control STATIC src/dynamics.cpp)
target_include_directories(rtforge_control PUBLIC include)

add_executable(rtforge_selftest tools/selftest.cpp)
target_link_libraries(rtforge_selftest PRIVATE rtforge_control)

add_executable(rtforge_track tools/track.cpp)
target_link_libraries(rtforge_track PRIVATE rtforge_control)

enable_testing()
add_test(NAME selftest COMMAND rtforge_selftest)
