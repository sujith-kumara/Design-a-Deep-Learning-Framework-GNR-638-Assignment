cmake_minimum_required(VERSION 3.10)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying to fetch it...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create the Python module (use different target name than C++ lib)
pybind11_add_module(deeplearn_py python_bindings.cpp)

# Link against the C++ deeplearn library
target_link_libraries(deeplearn_py PRIVATE deeplearn)

# Set output name to 'deeplearn' (creates deeplearn.so/.dylib)
set_target_properties(deeplearn_py PROPERTIES OUTPUT_NAME deeplearn)

# Install the module
install(TARGETS deeplearn_py
        LIBRARY DESTINATION ${Python3_SITELIB})
