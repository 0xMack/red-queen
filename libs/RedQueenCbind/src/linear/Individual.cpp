//
// Created by mackp on 2023-01-13.
//

#include "Individual.h"
#include <cstdint>
#include <iostream>
#include <bitset>
#include <sstream>
#include "Utils.h"


using namespace std;

unsigned short generateRandomShort() {
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_int_distribution<unsigned short> dis;
    return dis(gen);
}



Individual::Individual(int id, GrayCoder *grayCoder, vector<FloatOperation> *operators, float mutationRate) {
    ops = *operators;
    this->mutationRate = mutationRate;
    numOps = ops.size();
    for (int i = 0; i < numInstructions; i++) {
        program.emplace_back();
    }
    reset();
}

void Individual::reset() {
    fill_n(registers, numRegisters, 1);
};

void Individual::predict(vector<float>& inputData, vector<float>& outputData) {
    reset();
    for (auto &instruction: program) {
        executeInstruction(&instruction, &inputData[0], inputData.size());
    }

    copy(&registers[0], &registers[numRegisters], outputData.begin());
}

void Individual::predict(vector<vector<float>>& inputData, vector<vector<float>>& outputData) {
    auto t = inputData.size();
    for (auto i = 0; i < t; i++) {
        predict(inputData[i], outputData[i]);
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
