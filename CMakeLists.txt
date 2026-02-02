cmake_minimum_required(VERSION 3.10)
project(DeepLearnFramework)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(framework/include)

# Skeleton library
add_library(deeplearn 
    framework/src/tensor.cpp
    framework/src/layers.cpp
    framework/src/optim.cpp
)

# Subdirectories
add_subdirectory(tests)
# add_subdirectory(bindings)
