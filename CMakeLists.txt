cmake_minimum_required(VERSION 3.10)
project (langops)
add_definitions(-std=c++17)

# Configuration
set(LUA_VERSION 5.4.4)
set(ONNXRUNTIME_VERSION 1.13.1)
set(ONNXRUNTIME_ARCH linux-x64)

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

# Download an ONNX Runtime release to link against
set(ONNXRUNTIME_RELEASE_NAME onnxruntime-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION})
set(ONNXRUNTIME_RELEASE_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_RELEASE_NAME}.tgz)
file(DOWNLOAD
    ${ONNXRUNTIME_RELEASE_URL}
    ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz)
file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/external/onnxruntime.tgz DESTINATION ${CMAKE_BINARY_DIR}/external/)
set(ONNXRUNTIME_DIR ${CMAKE_BINARY_DIR}/external/${ONNXRUNTIME_RELEASE_NAME})

# Add header files & libraries
include_directories(${ONNXRUNTIME_DIR}/include/)
link_directories(${ONNXRUNTIME_DIR}/lib/)

# Find libraries we're linking against
find_library(LUA_LIBRARY lua HINTS ${LUA_DIR}/install/lib/)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib/)

# Set up target
add_library(langops SHARED lang_ops.cpp lang_ops.hpp lua_op.cpp lua_op.hpp)
target_link_libraries(langops PUBLIC ${ONNXRUNTIME_LIBRARY} ${LUA_LIBRARY})
