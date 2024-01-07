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

uint8_t generateRandomUint8();

struct Instruction {
    uint8_t inputSelector = generateRandomUint8();
    uint8_t arg1 = generateRandomUint8();
    uint8_t op = generateRandomUint8();
    uint8_t arg2 = generateRandomUint8();
    uint8_t output = generateRandomUint8();
};

class Individual {
    int id{};
    int numInstructions = 16;
    float mutationRate = 0.1;
    int numRegisters = N_REGISTERS;
    float registers[N_REGISTERS];
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
