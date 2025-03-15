#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include "train_model.hpp"  // Include TrainModel header

// Forward declaration for GenreModel if it's defined in another file
class GenreModel;

class Classifier {
public:
    // Constructor to use the shared model
    Classifier(TrainModel& model);
    
    // Method to classify text directly (without file)
    std::string classifyText(const std::string& text);
    
private:
    // Shared model containing genre models
    TrainModel& sharedModel;
    
    // Helper methods for preprocessing and calculating probabilities
    std::vector<std::string> preprocessText(const std::string& text);
    double calculateLogProbability(const std::vector<std::string>& words, const GenreModel& genreModel);
};

#endif // CLASSIFIER_HPP
