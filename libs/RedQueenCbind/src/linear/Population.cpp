//
// Created by mackp on 2023-01-13.
//
#include "Population.h"
#include <cstdio>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Individual.h"
namespace py = pybind11;

using namespace std;

Population::Population(vector<vector<float>>* dataset) {
    auto grayCoder = GrayCoder(16);
    vector<FloatOperation> ops = {add, sub, multiply};
    data = *dataset;
    auto popSize = 32;
    for (int i = 0; i < popSize; i++) {
        auto ind = Individual(i, &grayCoder, &ops, 0.1);
        individuals.push_back(ind);
    }

    printf("Initialized population of size: %d\n", popSize);
}

vector<vector<vector<float>>> Population::predict() {
    vector<vector<vector<float>>> all_predictions;
    all_predictions.reserve(individuals.size());
    for (auto individual: individuals){
        int size = data.size();
        vector<vector<float>> outputData(size, vector<float>(numRegisters));
        individual.predict(data, outputData);
        all_predictions.push_back(outputData);
    }
    return all_predictions;
}

template<typename T>
py::array_t<T> vector3d_to_numpy(const std::vector<std::vector<std::vector<T>>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        return py::array_t<T>({0, 0, 0});
    }

    size_t dim1 = vec.size();          // number of individuals
    size_t dim2 = vec[0].size();       // number of data points
    size_t dim3 = vec[0][0].size();    // number of registers

    std::vector<size_t> shape = {dim1, dim2, dim3};
    py::array_t<T> result(shape);
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);

    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                ptr[i * dim2 * dim3 + j * dim3 + k] = vec[i][j][k];
            }
        }
    }

    return result;
}


std::vector<std::vector<float>> numpy_to_vector2d(const py::array_t<float>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D NumPy array");

    size_t rows = buf.shape[0], cols = buf.shape[1];
    const auto* ptr = static_cast<float*>(buf.ptr);

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[i][j] = ptr[i * cols + j];

    return result;
}

// Helper function to convert vector to numpy array
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<std::vector<T>>& vec) {
    if (vec.empty()) {
        return py::array_t<T>({0, 0});
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    // Create a numpy array with the same shape
    py::array_t<T> result({rows, cols});
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    
    // Copy the data
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptr[i * cols + j] = vec[i][j];
        }
    }
    
    return result;
}

// Example usage in your PYBIND11_MODULE:
PYBIND11_MODULE(redqueen, m) {
    py::class_<Population>(m, "Population")
        .def(py::init([](const py::array_t<float>& arr) {
            auto arr2 = numpy_to_vector2d(arr);
            return new Population(&arr2);
        }))
        .def(py::init<std::vector<std::vector<float>> *>())
        // .def("predict", &Population::predict)
        // Example of returning a numpy array
        .def("predict", [](Population& self) -> py::array_t<float> {
            // Assuming you have a method that returns std::vector<std::vector<float>>
            vector<vector<vector<float>>> results = self.predict();
            return vector3d_to_numpy<float>(results);
        });
}