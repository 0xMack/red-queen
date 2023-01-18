//
// Created by mackp on 2023-01-13.
//

#ifndef REDQUEEN_INDIVIDUAL_H
#define REDQUEEN_INDIVIDUAL_H
#include <vector>
#include <cstdint>
#include "Utils.h"

using namespace std;


struct Instruction {
    uint8_t inputSelector = 0x0;
    uint8_t arg1 = 0x0;
    uint8_t op = 0x0;
    uint8_t arg2 = 0x0;
    uint8_t output = 0x0;
};

class Individual {
    int id{};
    const static int numRegisters = 8;
    int numInstructions = 16;
    float registers[numRegisters]{};
    GrayCoder *grayCoder{};
    vector<FloatOperation> ops{};
    int numOps;
    vector<Instruction> program = {};

public:
    explicit Individual(int id, GrayCoder *grayCoder, vector<FloatOperation> *operators);
    void runProgram(vector<float> *inputData);
    void reset();
private:
    void executeInstruction(Instruction *instruction, float* inputData, int inputSize);

};


#endif //REDQUEEN_INDIVIDUAL_H
