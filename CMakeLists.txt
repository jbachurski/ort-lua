cmake_minimum_required(VERSION 3.10)
project (langops)
add_definitions(-std=c++17)

# Configuration
set(LUA_VERSION 5.4.4)
set(ONNXRUNTIME_VERSION 1.13.1)

set(ONNXRUNTIME_INCLUDE_URL
    https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session)

# Download a Lua release
set(LUA_SOURCE_URL https://www.lua.org/ftp/lua-${LUA_VERSION}.tar.gz)
file(DOWNLOAD
    https://www.lua.org/ftp/lua-5.4.4.tar.gz
    ${CMAKE_BINARY_DIR}/external/lua.tgz)
file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/external/lua.tgz DESTINATION ${CMAKE_BINARY_DIR}/external/)
set(LUA_DIR ${CMAKE_BINARY_DIR}/external/lua-${LUA_VERSION}/)

# Build Lua
execute_process(COMMAND make all MYCFLAGS=-fPIC WORKING_DIRECTORY ${LUA_DIR})
# Copy built files to install/
execute_process(COMMAND make local WORKING_DIRECTORY ${LUA_DIR})

# Add header files & libraries
include_directories(${LUA_DIR}/install/include/)
link_directories(${LUA_DIR}/install/lib/)

# Download ONNX Runtime header files for given version
set(ONNXRUNTIME_DIR ${CMAKE_BINARY_DIR}/external/onnxruntime)
file(DOWNLOAD
    ${ONNXRUNTIME_INCLUDE_URL}/onnxruntime_c_api.h
    ${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h)
file(DOWNLOAD
    ${ONNXRUNTIME_INCLUDE_URL}/onnxruntime_cxx_inline.h
    ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_inline.h)
file(DOWNLOAD
    ${ONNXRUNTIME_INCLUDE_URL}/onnxruntime_cxx_api.h
    ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h)


# Add header files & libraries
include_directories(${ONNXRUNTIME_DIR}/include/)

# Find libraries we're linking against
find_library(LUA_LIBRARY lua HINTS ${LUA_DIR}/install/lib/)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib/)

# Set up target
add_library(langops SHARED lang_ops.cpp lang_ops.hpp lua_op.cpp lua_op.hpp)
target_link_libraries(langops PUBLIC ${LUA_LIBRARY})
