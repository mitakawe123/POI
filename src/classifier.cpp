#include "classifier.hpp"
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>

// Static instance pointer
Classifier* Classifier::instance = nullptr;
TrainModel* Classifier::sharedModel = nullptr;

// Private constructor to use the shared model
Classifier::Classifier(TrainModel& model) {
    sharedModel = &model; // Correctly initialize the static sharedModel
    std::cout << "[DEBUG] Classifier initialized with shared model." << std::endl;
    std::cout << "[DEBUG] Total genre models in shared model: " << sharedModel->genreModels.size() << std::endl;
}

// Public static method to get the singleton instance
Classifier& Classifier::getInstance() {
    if (instance == nullptr) {
        std::cerr << "[ERROR] Classifier not initialized yet. Please call initialize() first." << std::endl;
        throw std::runtime_error("Classifier not initialized.");
    }
    return *instance;
}

// Method to initialize the classifier with the model (only once)
void Classifier::initialize(TrainModel& model) {
    if (instance == nullptr) {
        instance = new Classifier(model);
    } else {
        std::cerr << "[ERROR] Classifier has already been initialized." << std::endl;
    }
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
    //std::cout << "[DEBUG] Calculating log probability for genre: " << genreModel.genre << std::endl;
   // std::cout << "[DEBUG] Genre Model Prior Probability: " << genreModel.priorProbability << std::endl;

    double logProbability = std::log(genreModel.priorProbability);
    //std::cout << "[DEBUG] Initial log probability (prior): " << logProbability << std::endl;

    // Iterate over words and update log probability based on word occurrences in the model
    for (const auto& word : words) {
        auto it = genreModel.wordProbabilities.find(word);
        if (it != genreModel.wordProbabilities.end()) {
            logProbability += std::log(it->second); // Log of word probability
            //std::cout << "[DEBUG] Word '" << word << "' found in model. Updated log probability: " << logProbability << std::endl;
        } else {
            // Smoothing applied when word is not found
            logProbability += std::log(1.0 / (genreModel.totalWordsInGenre + 1));  
            //std::cout << "[DEBUG] Word '" << word << "' NOT found. Applied smoothing. Updated log probability: " << logProbability << std::endl;
        }
    }

    std::cout << "[DEBUG] Final log probability for genre '" << genreModel.genre << "': " << logProbability << std::endl;
    return logProbability;
}

// Classify the text directly (without needing a file)
std::string Classifier::classifyText(const std::string& text) {
    std::cout << "[DEBUG] Starting text classification..." << std::endl;

    // Preprocess the input text
    std::vector<std::string> words = preprocessText(text);

    std::string bestGenre;
    double bestLogProbability = -std::numeric_limits<double>::infinity();

    std::cout << "[DEBUG] Evaluating " << sharedModel->genreModels.size() << " genre models." << std::endl;

    // Iterate through the genre models and calculate log probability for each genre
    for (const auto& genreEntry : sharedModel->genreModels) {
        const std::string& genre = genreEntry.first;
        const GenreModel& genreModel = genreEntry.second;

        std::cout << "[DEBUG] Checking genre: " << genre << std::endl;
        std::cout << "[DEBUG] Genre model details: " << std::endl;
        std::cout << "[DEBUG] Genre: " << genreModel.genre << ", Prior Probability: " << genreModel.priorProbability
                  << ", Total Words in Genre: " << genreModel.totalWordsInGenre << std::endl;

        double logProbability = calculateLogProbability(words, genreModel);

        // Update the best genre based on log probability comparison
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
