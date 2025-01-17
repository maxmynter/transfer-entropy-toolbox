#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fast_entropy {
    int add_numbers(int a , int b);
}

PYBIND11_MODULE(_fast_entropy, m){
    m.doc()= "Addition Hello world example.";
    m.def("add_numbers", &fast_entropy::add_numbers, "Add 2 nyumbers", py::arg("a"), py::arg("b"));
}
