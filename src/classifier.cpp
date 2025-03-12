#include "classifier.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>

using namespace std;

vector<string> Classify::preprocessText(const string& text) {
    vector<string> words;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        words.push_back(word);
    }
    return words;
}

void Classify::loadModel(const string& filename) {
    ifstream inFile(filename, ios::binary);
    while (!inFile.eof()) {
        string genre;
        char ch;
        while (inFile.get(ch) && ch != '\0') {
            genre += ch;
        }

        if (genre.empty()) break;

        GenreModel model;
        inFile.read(reinterpret_cast<char*>(&model.priorProbability), sizeof(model.priorProbability));
        inFile.read(reinterpret_cast<char*>(&model.totalWordsInGenre), sizeof(model.totalWordsInGenre));

        while (!inFile.eof()) {
            string word;
            while (inFile.get(ch) && ch != '\0') {
                word += ch;
            }

            if (word.empty()) break;

            double probability;
            inFile.read(reinterpret_cast<char*>(&probability), sizeof(probability));
            model.wordProbabilities[word] = probability;
        }

        genreModels[genre] = model;
    }
    inFile.close();
}

string Classify::classifyWithNaiveBayes(const string& summary) {
    vector<string> words = preprocessText(summary);
    string bestGenre;
    double bestScore = -INFINITY;

    for (const auto& genreEntry : genreModels) {
        string genre = genreEntry.first;
        const GenreModel& model = genreEntry.second;

        double score = log(model.priorProbability);

        for (const string& word : words) {
            if (model.wordProbabilities.find(word) != model.wordProbabilities.end()) {
                score += log(model.wordProbabilities.at(word));
            } else {
                score += log(1e-6); // Small probability for unseen words
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestGenre = genre;
        }
    }

    return bestGenre;
}
