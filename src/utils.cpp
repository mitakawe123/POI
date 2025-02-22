#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

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

void readFilesInDirectory() {
    for (const auto& filePath : fs::directory_iterator(Config::directoryPath)) {
        if(fs::is_regular_file(filePath)) {
            string file = filePath.path().string();
            readFileLineByLine(file);
        }
    }
}