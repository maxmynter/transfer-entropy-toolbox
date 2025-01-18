#include "discrete_entropy.h"
#include "multivar_entropy.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_fast_entropy, m) { // Make sure this matches your Python import
  m.doc() = "Fast entropy calculations using C++";

  m.def("discrete_entropy", &fast_entropy::discrete_entropy,
        "Calculate discrete entropy from class assignments",
        py::arg("data").noconvert(), py::arg("n_classes"));
  m.def("discrete_multivar_joint_entropy",
        &fast_entropy::discrete_multivar_joint_entropy,
        "Calculate discrete multivariate joint entropy",
        py::arg("data").noconvert(), py::arg("n_classes"));
}
