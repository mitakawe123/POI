#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <sstream>
#include <map>
#include <cmath>
#include <vector>
#include "config.hpp"
#include "utils.hpp"

using namespace std;
namespace fs = std::filesystem;


// string in C++ is value type that is why we are passing filename reference 
void readFileLineByLine(const string& filename) {
    ifstream file(filename);

    if(!file) {
        cerr << "Error opening file:" << filename << endl;
        return;
    }

    string line;
    while(getline(file, line)) {
        cout << line << endl;
    }
}

std::vector<std::string> readFilesInDirectory() {
    std::vector<std::string> files;

    for (const auto& filePath : fs::directory_iterator(Config::directoryPath)) {
        if (fs::is_regular_file(filePath)) {
            files.push_back(filePath.path().string());  // Add file path to vector
        }
    }

    return files;  // Return the vector of file paths
}

void processFile(const std::string& filePath, std::string& fileContent) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    fileContent = buffer.str();
    file.close();
}