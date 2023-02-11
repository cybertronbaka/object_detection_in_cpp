cmake_minimum_required(VERSION 3.17)
project(obj_detection)

set(CMAKE_CXX_STANDARD 17)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(
    "/usr/local/include/"
    "/usr/include/"
)

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION /usr/local/lib/libtensorflowlite.so)
set_property(TARGET  PROPERTY IMPORTED_LOCATION /usr/lib32/libpthread.so.0)

add_executable(obj_detection main.cpp)
target_link_libraries(
    obj_detection 
    tensorflowlite
    pthread
    opencv_core
    opencv_videoio
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)