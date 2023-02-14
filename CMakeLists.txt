cmake_minimum_required(VERSION 3.17)
project(obj_detection)

set(CMAKE_CXX_STANDARD 17)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(
    "./tflite-dist/include"
)

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION /tflite-dist/lib/libtensorflowlite.so)

add_executable(obj_detection main.cpp)
target_link_libraries(
    obj_detection 
    tensorflowlite
    opencv_core
    opencv_videoio
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)