cmake_minimum_required(VERSION 3.15)
project(fast_entropy)

# Ensure C++17 is required and enabled
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Basic optimization flag
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Create the module
pybind11_add_module(_fast_entropy
    src/bindings.cpp
    src/discrete_entropy.cpp
    src/discrete_multivar_entropy.cpp
    src/discrete_joint_entropy.cpp
)

# Add include directories
target_include_directories(_fast_entropy PRIVATE
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)
