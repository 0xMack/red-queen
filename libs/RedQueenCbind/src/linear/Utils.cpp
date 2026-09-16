//
// Created by mackp on 2023-01-14.
//

#include "Utils.h"
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <bitset>
#include <iostream>
#include <limits>

using namespace std;

float add(float v1, float v2) {
    return v1 + v2;
}
float sub(float v1, float v2) {
    return v1 - v2;
}
float multiply(float v1, float v2) {
    return v1 * v2;
}

template<typename T>
string toBinaryString(T x) {
    stringstream ss;
    ss << bitset<sizeof(T) * 8>(x);
    return ss.str();
}

GrayCoder::GrayCoder(int nBits) {
    buildGrayCode(nBits);
    buildReverseGrayCode(nBits);
}
unsigned int GrayCoder::toGrayCode(unsigned int v) {
    return grayCode[v];
}
unsigned int GrayCoder::fromGrayCode(unsigned int v) {
    return reverseGrayCode[v];
}

/**
 * Returns a gray-code vector of size n
 * This vector acts as a lookup table to convert a binary representation to one who's consecutive values
 * differ by only one bit. This is useful for providing the GA with a continuous problem space to explore.
 **/
void GrayCoder::buildGrayCode(int nBits) {
    for(int j =0; j<1<<nBits; j++) {
        grayCode.push_back(j^(j>>1));
    }
}
void GrayCoder::buildReverseGrayCode(int nBits) {
    if (nBits > 32) {
        throw invalid_argument("nBits must less than 32");
    }
    for(unsigned int i =0; i<1<<nBits; i++) {
        unsigned int x = i;
        for (unsigned int mask = std::numeric_limits<unsigned int>::digits / 2; mask; mask >>= 1) {
            x ^= x >> mask;
        }
        reverseGrayCode.push_back(x);
    }
};
void GrayCoder::printCodes() {
    cout << "Gray Codes: [";
    for(unsigned int i : grayCode){
        cout << i << ", ";
    }
    cout << "]"<<endl;
    cout << "Rev. Codes: [";
    for(unsigned int i : reverseGrayCode){
        cout << i << ", ";
    }
    cout << "]"<<endl;
}

void test() {
    auto coder = GrayCoder(8);
    coder.printCodes();
    for (uint8_t i = 0; i < 32; i++) {

        uint8_t x = coder.toGrayCode(i);
        uint8_t y = coder.fromGrayCode(x);

        printf("%d --> %d --> %d\n", i, x, y);
        printf("%s --> %s --> %s\n",
               toBinaryString(i).c_str(),
        toBinaryString(x).c_str(),
        toBinaryString(y).c_str());
    }
}
