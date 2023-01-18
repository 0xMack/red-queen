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
        auto ind = Individual(i, &grayCoder, &ops);
        individuals.push_back(ind);
    }

    printf("Initialized population of size: %d\n", popSize);
}

void Population::evaluate() {
    for (auto i: individuals){
        i.runProgram(&data[0]);
    }
}