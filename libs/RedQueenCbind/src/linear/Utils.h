//
// Created by mackp on 2023-01-14.
//

#ifndef REDQUEEN_UTILS_H
#define REDQUEEN_UTILS_H

#include <cstdint>
#include <vector>
#include <string>

using namespace std;

typedef float (*FloatOperation) (float v1, float v2);

float add(float v1, float v2);
float sub(float v1, float v2);
float multiply(float v1, float v2);

template<typename T>
string toBinaryString(T x);

void test();



class Utils {

};

class GrayCoder {
    vector<unsigned int> grayCode;
    vector<unsigned short> reverseGrayCode;
public:
    explicit GrayCoder(int nBits);
    unsigned int toGrayCode(unsigned int v);
    unsigned int fromGrayCode(unsigned int v);
    void printCodes();
private:
    void buildGrayCode(int nBits);
    void buildReverseGrayCode(int nBits);
};

#endif //REDQUEEN_UTILS_H
