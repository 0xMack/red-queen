//
// Created by mackp on 2023-01-13.
//

#include "Main.h"
#include "DatasetLoader.h"
#include "Population.h"
#include <iostream>

int main(int argc, char *argv[]) {
    auto* dataLoader = new DatasetLoader("../data/iris.train.X", "../data/iris.train.Y");
    dataLoader->rawData;
    auto pop = Population(&dataLoader->rawData);
    pop.predict();
    return 0;
}
