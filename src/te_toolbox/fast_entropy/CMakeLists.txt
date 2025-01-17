cmake_minimum_required(VERSION 3.15)
project(fast_entropy)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fast_entropy
src/adder.cpp
src/bindings.cpp
)
