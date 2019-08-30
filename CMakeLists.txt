cmake_minimum_required(VERSION 2.6)
project(op)

set(CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp" )

FIND_PACKAGE(RTABMap REQUIRED)
include_directories(${RTABMap_INCLUDE_DIRS})
############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${RTABMap_LIBRARIES}
)

add_executable( op main.cpp )
target_link_libraries( op ${THIRD_PARTY_LIBS} )
