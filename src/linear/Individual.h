//
// Created by mackp on 2023-01-13.
//

#ifndef REDQUEEN_INDIVIDUAL_H
#define REDQUEEN_INDIVIDUAL_H
#include <vector>
#include <cstdint>
#include "Utils.h"
#include <random>

using namespace std;
#define N_REGISTERS 8

unsigned short generateRandomShort();

struct Instruction {
    unsigned short inputSelector = generateRandomShort();
    unsigned short arg1 = generateRandomShort();
    unsigned short op = generateRandomShort();
    unsigned short arg2 = generateRandomShort();
    unsigned short output = generateRandomShort();
};

class Individual {
    int id{};
    int numInstructions = 16;
    float mutationRate = 0.1;
    int numRegisters = N_REGISTERS;
    float registers[N_REGISTERS]{};
    GrayCoder *grayCoder{};
    vector<FloatOperation> ops{};
    int numOps;
    vector<Instruction> program = {};

public:
    explicit Individual(int id, GrayCoder *grayCoder, vector<FloatOperation> *operators, float mutationRate);
    void predict(vector<vector<float>>& inputData, vector<vector<float>>& outputData);
    void predict(vector<float>& inputData, vector<float>& outputData);
    void reset();
private:
    void executeInstruction(Instruction *instruction, float* inputData, int inputSize);

};


#endif //REDQUEEN_INDIVIDUAL_H
