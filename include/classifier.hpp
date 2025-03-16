#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <string>
#include <vector>
#include <queue>
#include <train_model.hpp>
#include <genre_model.hpp>

class Classifier {
public:
    // Delete copy constructor and assignment operator to ensure only one instance
    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;

    // Public static method to get the instance (without passing a model)
    static Classifier& getInstance();

    // Method to classify text
    std::string classifyText(const std::string& text);

    // Method to initialize the classifier with the model (only once)
    static void initialize(TrainModel& model);

private:
    // Private constructor to prevent instantiation outside of the class
    Classifier(TrainModel& model);

    // Static instance pointer for Singleton pattern
    static Classifier* instance;

    // Shared model used by the classifier (it is set only once via initialization)
    static TrainModel* sharedModel;

    // Helper methods for preprocessing and calculating log probabilities
    std::vector<std::string> preprocessText(const std::string& text);
    double calculateLogProbability(const std::vector<std::string>& words, const GenreModel& genreModel);
};

#endif // CLASSIFIER_HPP
