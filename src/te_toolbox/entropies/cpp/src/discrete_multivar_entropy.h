#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace fast_entropy {
double discrete_multivar_joint_entropy(
    const std::vector<py::array_t<int64_t>> &classes,
    const std::vector<int> &n_classes);
} // namespace fast_entropy
