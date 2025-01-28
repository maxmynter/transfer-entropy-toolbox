#include "discrete_conditional_entropy.h"
#include "discrete_entropy.h"
#include "discrete_joint_entropy.h"
#include "discrete_multivar_entropy.h"
#include "discrete_transfer_entropy.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_fast_entropy, m) {
  m.doc() = "Fast entropy calculations using C++";

  m.def("discrete_entropy", &fast_entropy::discrete_entropy,
        "Calculate discrete entropy from class assignments",
        py::arg("data").noconvert(), py::arg("n_classes"));

  m.def("discrete_joint_entropy", &fast_entropy::discrete_joint_entropy,
        "Calculate bivariate discrete joint entropy",
        py::arg("data").noconvert(), py::arg("n_classes"));

  m.def("discrete_multivar_joint_entropy",
        &fast_entropy::discrete_multivar_joint_entropy,
        "Calculate discrete multivariate joint entropy",
        py::arg("data").noconvert(), py::arg("n_classes"));

  m.def("discrete_conditional_entropy",
        &fast_entropy::discrete_conditional_entropy,
        "Calculate conditional entropy H(Y|X) between the two column variables "
        "[X,Y]",
        py::arg("data").noconvert(), py::arg("n_classes"));

  m.def("discrete_transfer_entropy", &fast_entropy::discrete_transfer_entropy,
        "Calculate transfer entropy from X to Y given time series [Y,X]",
        py::arg("data").noconvert(), py::arg("n_classes"), py::arg("lag"));
}
