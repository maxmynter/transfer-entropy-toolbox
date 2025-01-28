#pragma once
#include "discrete_entropy.h"
#include "discrete_joint_entropy.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace fast_entropy {
double discrete_conditional_entropy(const py::array_t<int64_t> &data,
                                    const std::vector<int> &n_classes);
} // namespace fast_entropy
