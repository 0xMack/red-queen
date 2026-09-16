//
// Created by mackp on 2023-01-13.
//

#ifndef REDQUEEN_DATASETLOADER_H
#define REDQUEEN_DATASETLOADER_H

#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

using namespace std;

class DatasetLoader {
public:
    vector<vector<float>> rawData;
    vector<int> labels;

    explicit DatasetLoader(const string& dataFilepath, const string& labelFilepath) {
        if (!fileExists(dataFilepath)) {
            throw invalid_argument("Dataset file does not exist. " + dataFilepath);
        }
        loadData(dataFilepath);
        loadLabels(labelFilepath);
    }

private:
    static bool fileExists(const string &fileName) {
        std::ifstream infile(fileName);
        return infile.good();
    }

    void loadData(const string &fileName) {
        vector<float> row;
        string line, word;
        fstream file(fileName, ios::in);
        if (file.is_open()) {
            while (getline(file, line)) {
                row.clear();
                stringstream str(line);
                while (getline(str, word, ','))
                    row.push_back(stof(word));
                rawData.push_back(row);
            }
        }
    }
    void loadLabels(const string &fileName) {
        string line, word;
        fstream file(fileName, ios::in);
        if (file.is_open()) {
            while (getline(file, line)) {
                stringstream str(line);
                while (getline(str, word, ','))
                    labels.push_back(stoi(word));
            }
        }
    }

};


#endif //REDQUEEN_DATASETLOADER_H
