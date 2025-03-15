#ifndef TRAIN_MODEL_HPP
#define TRAIN_MODEL_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include "genre_model.hpp"

class TrainModel {
public:
    TrainModel();
    void trainOrLoadModel(const std::string& modelFilename);
    void trainNaiveBayes();
    void saveModel(const std::string& filename);
    void displayModel() const;
    void loadModel(const std::string& filename);
    std::unordered_map<std::string, GenreModel> genreModels;  
    void addGenreModel(const std::string& genre, GenreModel& genreModel);

private:
    std::vector<std::string> preprocessText(const std::string& text);
    std::vector<std::pair<std::string, std::string>> readCSV(const std::string& fileName);

private:
    int totalDocuments;
};

#endif // TRAIN_MODEL_HPP
