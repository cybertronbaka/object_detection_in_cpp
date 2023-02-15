cmake_minimum_required(VERSION 3.17)
project(obj_detection)

set(CMAKE_CXX_STANDARD 17)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/tflite-dist/include/
)

#link_directories(/path/to/library/directory)
# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
ADD_LIBRARY(opencv_core SHARED IMPORTED)
ADD_LIBRARY(opencv_videoio SHARED IMPORTED)
ADD_LIBRARY(opencv_highgui SHARED IMPORTED)
ADD_LIBRARY(opencv_imgcodecs SHARED IMPORTED)
ADD_LIBRARY(opencv_imgproc SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libtensorflowlite.so)
set_property(TARGET opencv_core PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libopencv_core.so)
set_property(TARGET opencv_videoio PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libopencv_videoio.so)
set_property(TARGET opencv_highgui PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libopencv_highgui.so)
set_property(TARGET opencv_imgcodecs PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libopencv_imgcodecs.so)
set_property(TARGET opencv_imgproc PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tflite-dist/lib/libopencv_imgproc.so)
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
