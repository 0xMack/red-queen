//
// Created by mackp on 2023-02-11.
//
#include <pybind11/pybind11.h>

namespace py = pybind11;


int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(redqueen, m) {
    m.def("add", &add, "A function that adds two numbers");
}
