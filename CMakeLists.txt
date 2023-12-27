cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)



find_package(OpenCV REQUIRED)

include_directories("include/")
include_directories(/date2/wangzijian/onnxruntime-linux-x64-gpu-1.12.1/include)


add_executable(yolo_ort
               src/main.cpp
               src/detector.cpp
               src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(yolo_ort PRIVATE cxx_std_14)
target_link_libraries(yolo_ort ${OpenCV_LIBS})


target_link_libraries(yolo_ort /date2/wangzijian/onnxruntime-linux-x64-gpu-1.12.1/lib/libonnxruntime.so)


