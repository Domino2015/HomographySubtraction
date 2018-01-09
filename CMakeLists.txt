cmake_minimum_required(VERSION 3.9)
project(HomographySubtraction)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(HomographySubtraction main.cpp)
target_link_libraries(HomographySubtraction ${OpenCV_LIBS} )