#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace fast_entropy {

double discrete_entropy(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> data,
    int n_classes);

}
