#include "classifier.hpp"
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <genre_model.hpp>
#include <train_model.hpp>

// Constructor to use the shared model
Classifier::Classifier(TrainModel& model) : sharedModel(model) {
    std::cout << "[DEBUG] Classifier initialized with shared model." << std::endl;
    std::cout << "[DEBUG] Total genre models in shared model: " << sharedModel.genreModels.size() << std::endl;
}

// Preprocess the text (convert to lowercase and remove punctuation)
std::vector<std::string> Classifier::preprocessText(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;

    std::cout << "[DEBUG] Preprocessing text..." << std::endl;
    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        words.push_back(word);
    }

    std::cout << "[DEBUG] Preprocessed text contains " << words.size() << " words." << std::endl;
    return words;
}

// Calculate log probability for the text given the genre model
double Classifier::calculateLogProbability(const std::vector<std::string>& words, const GenreModel& genreModel) {
    std::cout << "[DEBUG] Calculating log probability for genre: " << genreModel.genre << std::endl;
    std::cout << "[DEBUG] Genre Model Prior Probability: " << genreModel.priorProbability << std::endl;

    double logProbability = std::log(genreModel.priorProbability);
    std::cout << "[DEBUG] Initial log probability (prior): " << logProbability << std::endl;

    for (const auto& word : words) {
        auto it = genreModel.wordProbabilities.find(word);
        if (it != genreModel.wordProbabilities.end()) {
            logProbability += std::log(it->second);
            std::cout << "[DEBUG] Word '" << word << "' found in model. Updated log probability: " << logProbability << std::endl;
        } else {
            logProbability += std::log(1.0 / (genreModel.totalWordsInGenre + 1));  // Smoothing
            std::cout << "[DEBUG] Word '" << word << "' NOT found. Applied smoothing. Updated log probability: " << logProbability << std::endl;
        }
    }

    std::cout << "[DEBUG] Final log probability for genre '" << genreModel.genre << "': " << logProbability << std::endl;
    return logProbability;
}

// Classify the text directly (without needing a file)
std::string Classifier::classifyText(const std::string& text) {
    std::cout << "[DEBUG] Starting text classification..." << std::endl;
    
    std::vector<std::string> words = preprocessText(text);

    std::string bestGenre;
    double bestLogProbability = -std::numeric_limits<double>::infinity();

    std::cout << "[DEBUG] Evaluating " << sharedModel.genreModels.size() << " genre models." << std::endl;
    
    for (const auto& genreEntry : sharedModel.genreModels) {
        const std::string& genre = genreEntry.first;
        const GenreModel& genreModel = genreEntry.second;
        
        std::cout << "[DEBUG] Checking genre: " << genre << std::endl;
        std::cout << "[DEBUG] Genre model details: " << std::endl;
        std::cout << "[DEBUG] Genre: " << genreModel.genre << ", Prior Probability: " << genreModel.priorProbability
                  << ", Total Words in Genre: " << genreModel.totalWordsInGenre << std::endl;

        double logProbability = calculateLogProbability(words, genreModel);

        if (logProbability > bestLogProbability) {
            bestLogProbability = logProbability;
            bestGenre = genre;
        }
    }

    if (bestGenre.empty()) {
        std::cerr << "[ERROR] Classification failed: No valid genre found." << std::endl;
        return "Unknown";
    }

    std::cout << "[INFO] Text classified as: " << bestGenre << " with log probability: " << bestLogProbability << std::endl;
    return bestGenre;
}
