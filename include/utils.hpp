// utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <map>

std::vector<std::string> readFilesInDirectory();
void readFileLineByLine(const std::string& filename);  // Declare function for reading a file line by line

// Naive Bayes classifier functions
void trainNaiveBayes(const std::vector<std::pair<std::string, std::string>>& labeledData);
std::string classifyWithNaiveBayes(const std::string& fileContent);

#endif
