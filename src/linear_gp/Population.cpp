//
// Created by mackp on 2023-01-13.
//
#include <cstdio>
#include "Population.h"
#include "Individual.h"

using namespace std;

Population::Population(vector<vector<float>>* dataset) {
    auto grayCoder = GrayCoder(8);
    vector<FloatOperation> ops = {add, sub, multiply};
    data = *dataset;
    auto popSize = 32;
    for (int i = 0; i < popSize; i++) {
        auto ind = Individual(i, &grayCoder, &ops, 0.1);
        individuals.push_back(ind);
    }

    printf("Initialized population of size: %d\n", popSize);
}

void Population::predict() {

    for (auto i: individuals){
        int size = data.size();
        vector<vector<float>> outputData(size, vector<float>(numRegisters));
        i.predict(data, outputData);
    }
}

//
//namespace py = pybind11;
//
//PYBIND11_MODULE(redqueen, m) {
//    py::class_<Population>(m, "Population")
//    .def(py::init<std::vector<std::vector<float>> *>())
//    .def("predict", &Population::predict);
//}