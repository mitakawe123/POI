#include "train_model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace std;

vector<string> TrainModel::preprocessText(const string& text) {
    vector<string> words;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        // Convert each word to lowercase and remove non-alphanumeric characters
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(remove_if(word.begin(), word.end(), [](char c) { return !isalnum(c); }), word.end());
        words.push_back(word);
    }
    return words;
}

vector<pair<string, string>> TrainModel::readCSV(const string& fileName) {
    vector<pair<string, string>> data;
    ifstream file(fileName);

    // Check if the file is open
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << fileName << endl;
        return data;
    }

    string line;
    getline(file, line); // Skip the header line
    while (getline(file, line)) {
        stringstream ss(line);
        string title, genre, summary;
        
        // Read the genre and summary, skipping the title
        getline(ss, title, ',');
        getline(ss, genre, ',');
        getline(ss, summary);
        
        // Ensure we don't push empty values
        if (!genre.empty() && !summary.empty()) {
            data.push_back(make_pair(genre, summary));
        }
    }
    file.close();
    return data;
}

void TrainModel::trainNaiveBayes() {
    vector<pair<string, string>> trainingData = readCSV("../book.csv");  // Always use book.csv here
    unordered_map<string, int> genreDocumentCounts;
    unordered_map<string, unordered_map<string, int>> tempWordCounts;

    // Process each record from the CSV
    for (const auto& record : trainingData) {
        string genre = record.first;
        string summary = record.second;
        vector<string> words = preprocessText(summary);

        // Count the number of documents per genre
        genreDocumentCounts[genre]++;
        totalDocuments++;

        // Count the word frequencies for each genre
        for (const string& word : words) {
            tempWordCounts[genre][word]++;
        }
    }

    // Calculate probabilities for each genre
    for (const auto& genreEntry : tempWordCounts) {
        string genre = genreEntry.first;
        GenreModel model;
        model.totalWordsInGenre = 0;

        // Calculate the word probabilities for each genre
        for (const auto& wordEntry : genreEntry.second) {
            model.wordProbabilities[wordEntry.first] = static_cast<double>(wordEntry.second);
            model.totalWordsInGenre += wordEntry.second;
        }

        // Calculate the prior probability for the genre
        model.priorProbability = static_cast<double>(genreDocumentCounts[genre]) / totalDocuments;

        // Normalize word probabilities
        for (auto& wordEntry : model.wordProbabilities) {
            wordEntry.second /= model.totalWordsInGenre;
        }

        genreModels[genre] = model;
    }
}

void TrainModel::saveModel(const string& filename) {
    ofstream outFile(filename, ios::binary);

    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // Write each genre model to the binary file
    for (const auto& genreEntry : genreModels) {
        outFile.write(genreEntry.first.c_str(), genreEntry.first.size());
        outFile.put('\0');  // Null-terminate the genre name
        outFile.write(reinterpret_cast<const char*>(&genreEntry.second.priorProbability), sizeof(genreEntry.second.priorProbability));
        outFile.write(reinterpret_cast<const char*>(&genreEntry.second.totalWordsInGenre), sizeof(genreEntry.second.totalWordsInGenre));

        // Write each word's probability
        for (const auto& wordEntry : genreEntry.second.wordProbabilities) {
            outFile.write(wordEntry.first.c_str(), wordEntry.first.size());
            outFile.put('\0');  // Null-terminate the word
            outFile.write(reinterpret_cast<const char*>(&wordEntry.second), sizeof(wordEntry.second));
        }
    }
    outFile.close();
}
