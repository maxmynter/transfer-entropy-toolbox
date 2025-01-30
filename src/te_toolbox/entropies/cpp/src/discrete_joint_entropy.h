#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace fast_entropy {
double discrete_joint_entropy(const py::array_t<int64_t> &data,
                              const std::vector<int> &n_classes);
}
