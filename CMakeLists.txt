cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++17)

set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.cc)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

file(DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-osx-arm64-1.13.1.tgz
    ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz)
file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz)

#Include path to header files for Custom Op
include_directories(<PATH_TO_ONNXRUNTIME_INCLUDE_DIR>)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS <PATH_TO_ONNXRUNTIME_LIB>)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})