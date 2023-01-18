//
// Created by mackp on 2023-01-13.
//

#include "Individual.h"
#include <cstdint>
#include <iostream>
#include <bitset>
#include <sstream>
#include <Utils.h>

using namespace std;


Individual::Individual(int id, GrayCoder *grayCoder, vector<FloatOperation> *operators) {
    grayCoder = grayCoder;
    id = id;
    ops = *operators;
    numOps = ops.size();
    for (int i = 0; i < numInstructions; i++) {
        auto inst = new Instruction();
        program.push_back(*inst);
    }
    reset();
}

void Individual::reset() {
    fill_n(registers, numRegisters, 1);
};

void Individual::runProgram(vector<float> *inputData) {
    for (auto i: program) {
        auto data = *inputData;
        executeInstruction(&i, &data[0],data.size());
    }
}

void Individual::executeInstruction(Instruction *instruction, float *inputData, int inputSize) {
    int sourceSize;
    float *source;
    if (instruction->inputSelector % 2) {
        source = registers;
        sourceSize = numRegisters;
    } else {
        source = inputData;
        sourceSize = inputSize;
    }
    auto arg1Idx = instruction->arg1 % numRegisters;
    auto opIdx = instruction->op % numOps;
    auto arg2Idx = instruction->arg2 % sourceSize;
    auto outputIdx = instruction->output % numRegisters;
    registers[outputIdx] = ops[opIdx](registers[arg1Idx], source[arg2Idx]);
}