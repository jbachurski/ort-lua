cmake_minimum_required(VERSION 3.10)
project (luaop)
add_definitions(-std=c++17)

set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.cc)
set(ONNXRUNTIME_VERSION 1.13.1)
set(ONNXRUNTIME_ARCH osx-arm64)
add_library(luaop SHARED ${SOURCE} ${HEADER})

set(ONNXRUNTIME_RELEASE_NAME onnxruntime-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION})
set(ONNXRUNTIME_RELEASE_LINK https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_RELEASE_NAME}.tgz)
# Download the right release architecture here!
file(DOWNLOAD
    ${ONNXRUNTIME_RELEASE_LINK}    
    ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz)
file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz DESTINATION ${CMAKE_BINARY_DIR}/external/)
set(ONNXRUNTIME_DIR ${CMAKE_BINARY_DIR}/external/${ONNXRUNTIME_RELEASE_NAME})

#Include path to header files for Custom Op
include_directories(${ONNXRUNTIME_DIR}/include/)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib/)
target_link_libraries(luaop PUBLIC ${ONNXRUNTIME_LIBRARY})