#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <unordered_map>
#include <vector>
#include <string>

struct GenreModel {
    std::unordered_map<std::string, double> wordProbabilities;
    double priorProbability;
    int totalWordsInGenre;
};

class Classify {
public:
    void loadModel(const std::string& filename);
    std::string classifyWithNaiveBayes(const std::string& summary);
    
private:
    std::unordered_map<std::string, GenreModel> genreModels;
    std::vector<std::string> preprocessText(const std::string& text);
};

#endif
