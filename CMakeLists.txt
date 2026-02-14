cmake_minimum_required(VERSION 3.10)
project(DeepLearnFramework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# Options
option(USE_CUDA "Build with CUDA support" OFF)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support disabled (CPU only)")
endif()

# Include directories
include_directories(framework/include)

# Skeleton library
set(SOURCES
    framework/src/tensor.cpp
    framework/src/layers.cpp
    framework/src/optim.cpp
)

if(USE_CUDA)
    # Add .cu files here when they exist
    # list(APPEND SOURCES framework/src/tensor_cuda.cu)
endif()

add_library(deeplearn ${SOURCES})

# Subdirectories
add_subdirectory(bindings)
