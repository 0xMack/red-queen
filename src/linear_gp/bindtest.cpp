//
// Created by mackp on 2023-02-11.
//
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

double testBind(int a) {
    return a * 0.1;
}

namespace py = pybind11;

PYBIND11_MODULE(redqueen, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("testBind", &testBind, R"pbdoc(
            A python function that returns a result from c code
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
