//
// Created by mackp on 2023-01-13.
//

#ifndef REDQUEEN_POPULATION_H
#define REDQUEEN_POPULATION_H


#include <cstdint>
#include <vector>
#include "Utils.h"
#include "Individual.h"

using namespace std;

class Population {
    vector<Individual> individuals;
    vector<vector<float>> data;
    int size{};
    int numRegisters{};
    float mutationRate{};
    float crossoverRate{};

public:
    explicit Population(vector<vector<float>>* dataset);

    void evaluate();
};


#endif //REDQUEEN_POPULATION_H
