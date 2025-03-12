#ifndef TRAIN_MODEL_HPP
#define TRAIN_MODEL_HPP

#include <unordered_map>
#include <vector>
#include <string>

struct GenreModel {
    std::unordered_map<std::string, double> wordProbabilities;
    double priorProbability;
    int totalWordsInGenre;
};

class TrainModel {
public:
    void trainNaiveBayes();
    void saveModel(const std::string& filename);
    
private:
    std::unordered_map<std::string, GenreModel> genreModels;
    int totalDocuments = 0;
    std::vector<std::pair<std::string, std::string>> readCSV(const std::string& fileName);
    std::vector<std::string> preprocessText(const std::string& text);
};

#endif
