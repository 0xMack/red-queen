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

double testBind(int a);

class Population {
    vector<Individual> individuals;
    vector<vector<float>> data;
    int size{};
    int numRegisters{};
    float mutationRate{};
    float crossoverRate{};

public:
    Population(vector<vector<float>> *dataset);

    void predict();
};

extern "C" {
//  Population* LGP_Population(vector<vector<float>> *dataset){ return new Population(dataset);}
//  void predict();
};

#endif //REDQUEEN_POPULATION_H
