cmake_minimum_required(VERSION 3.5)
project(retinaface_ncnn)
set(TARGET_NAME "retinaface_ncnn")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pie -fPIE -fPIC -O3")
set(SOURCE_FILES src/anchor_generator.cpp  src/config.cpp  src/detect.cpp  src/tools.cpp
        src/anchor_generator.h src/config.h src/tools.h)

include_directories("/usr/local/opencv-3.1.0/include")
link_directories("/usr/local/opencv-3.1.0/build/lib")
set(OpenCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_calib3d opencv_video opencv_videoio opencv_objdetect opencv_ml opencv_features2d opencv_photo opencv_flann opencv_imgcodecs)
#find_package(OpenCV REQUIRED)

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
link_directories("${CMAKE_CURRENT_LIST_DIR}/lib")

add_executable(retinaface_ncnn ${SOURCE_FILES})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} tvm_runtime dl pthread)
