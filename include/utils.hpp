// utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <map>

std::vector<std::string> readFilesInDirectory();
void readFileLineByLine(const std::string& filename);  // Declare function for reading a file line by line

void processFile(const std::string& filePath, std::string& fileContent);

#endif
