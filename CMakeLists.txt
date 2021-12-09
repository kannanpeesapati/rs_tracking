
cmake_minimum_required( VERSION 3.5.1 ) 
set (CMAKE_CXX_STANDARD 11)

project( rs_tracking ) 
 
 

find_package( OpenCV REQUIRED )
 

include_directories( ${OpenCV_INCLUDE_DIRS})
 

add_executable( rs_tracking rs_tracking.cpp ) 
target_link_libraries( rs_tracking ${OpenCV_LIBS} opencv_tracking) 


find_package(OpenCV)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

 
 




