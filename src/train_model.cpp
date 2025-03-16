#include "train_model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <filesystem>
#include <locale>
#include <codecvt>

using namespace std;
namespace fs = filesystem;

TrainModel::TrainModel() : totalDocuments(0) {}

vector<string> TrainModel::preprocessText(const string& text) {
    vector<string> words;
    stringstream ss(text);
    string word;

    while (ss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(remove_if(word.begin(), word.end(), [](char c) { return !isalnum(c); }), word.end());
        words.push_back(word);
    }
    return words;
}

vector<pair<string, string>> TrainModel::readCSV(const string& fileName) {
    vector<pair<string, string>> rows;
    wifstream file(fileName);
    wstring line;

    file.imbue(locale(locale::classic(), new codecvt_utf8<wchar_t>()));

    if (!file.is_open()) {
        cerr << "Error opening file: " << fileName << endl;
        return rows;
    }

    getline(file, line);  // Skip header

    while (getline(file, line)) {
        wstringstream ss(line);
        wstring dummy;
        getline(ss, dummy, L',');
        getline(ss, dummy, L',');

        wstring genre;
        getline(ss, genre, L',');

        if (genre.empty()) continue;

        wstring summary;
        bool inSummary = true;
        wstring summaryLine;
        getline(ss, summaryLine);
        summary += summaryLine;

        while (getline(file, line)) {
            if (line.find(L"(less)") != wstring::npos) break;
            if (line.empty()) continue;

            summary += L" " + line;
        }

        rows.push_back({string(genre.begin(), genre.end()), string(summary.begin(), summary.end())});
    }

    file.close();
    return rows;
}

void TrainModel::trainOrLoadModel(const string& modelFilename) {
    string projectDir = fs::current_path().string();  
    string fullPath = projectDir + "/" + modelFilename;  

    if (fs::exists(fullPath)) {
        loadModel(fullPath);
        cout << "Model loaded from file: " << modelFilename << endl;
    } else {
        trainNaiveBayes();
        saveModel(modelFilename);
        cout << "New model saved as: " << modelFilename << endl;
    }
}

void TrainModel::trainNaiveBayes() {
    vector<pair<string, string>> trainingData = readCSV("../extracted_book/output.csv");

    // Predefined list of genres to ensure they're included in the model
    vector<string> predefinedGenres = {
        "horror", "fantasy", "science", "crime", "history", 
        "thriller", "romance", "psychology", "sports", "travel"
    };

    // Add predefined genres to the model if they don't exist
    for (const auto& genre : predefinedGenres) {
        if (genreModels.find(genre) == genreModels.end()) {
            GenreModel newGenreModel;
            genreModels[genre] = newGenreModel;  // Initialize the genre with empty data
        }
    }

    unordered_map<string, int> genreDocumentCounts;
    unordered_map<string, unordered_map<string, int>> tempWordCounts;

    for (const auto& record : trainingData) {
        string genre = record.first;
        string summary = record.second;
        vector<string> words = preprocessText(summary);
        genreDocumentCounts[genre]++;
        totalDocuments++;

        // Count the word occurrences per genre
        for (const string& word : words) {
            tempWordCounts[genre][word]++;
        }
    }

    // Build the model for each genre
    for (const auto& genreEntry : tempWordCounts) {
        string genre = genreEntry.first;
        GenreModel model;
        model.totalWordsInGenre = 0;

        for (const auto& wordEntry : genreEntry.second) {
            model.wordProbabilities[wordEntry.first] = static_cast<double>(wordEntry.second);
            model.totalWordsInGenre += wordEntry.second;
        }

        model.priorProbability = static_cast<double>(genreDocumentCounts[genre]) / totalDocuments;
        for (auto& wordEntry : model.wordProbabilities) {
            wordEntry.second /= model.totalWordsInGenre;
        }

        genreModels[genre] = model;
    }

    displayModel();
}

void TrainModel::addGenreModel(const string& genre, GenreModel& genreModel) {
    genreModels[genre] = genreModel;
}

void TrainModel::saveModel(const string& filename) {
    string projectDir = fs::current_path().string();  
    string fullPath = projectDir + "/" + filename;

    if (!fs::exists(projectDir)) {
        if (fs::create_directory(projectDir)) {
            cout << "Created directory: " << projectDir << endl;
        } else {
            cerr << "Error: Failed to create directory: " << projectDir << endl;
            return;
        }
    }

    if (fs::exists(fullPath)) {
        cout << "File already exists: " << fullPath << ". Nothing to do." << endl;
        return;
    }

    ofstream outFile(fullPath, ios::binary);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << fullPath << " for writing." << endl;
        return;
    }

    cout << "Saving model to: " << fullPath << endl;

    for (const auto& genreEntry : genreModels) {
        outFile.write(genreEntry.first.c_str(), genreEntry.first.size());
        outFile.put('\0');
        outFile.write(reinterpret_cast<const char*>(&genreEntry.second.priorProbability), sizeof(genreEntry.second.priorProbability));
        outFile.write(reinterpret_cast<const char*>(&genreEntry.second.totalWordsInGenre), sizeof(genreEntry.second.totalWordsInGenre));

        for (const auto& wordEntry : genreEntry.second.wordProbabilities) {
            outFile.write(wordEntry.first.c_str(), wordEntry.first.size());
            outFile.put('\0');
            outFile.write(reinterpret_cast<const char*>(&wordEntry.second), sizeof(wordEntry.second));
        }
    }

    outFile.close();
    cout << "Model successfully saved to: " << fullPath << endl;

    displayModel();
}

void TrainModel::displayModel() const {
    for (const auto& genreEntry : genreModels) {
        cout << "Genre: " << genreEntry.first << endl;
        cout << "Prior Probability: " << genreEntry.second.priorProbability << endl;
        cout << "Total Words in Genre: " << genreEntry.second.totalWordsInGenre << endl;
        cout << "Top Word Probabilities:" << endl;

        vector<pair<string, double>> wordProbabilities;
        for (const auto& wordEntry : genreEntry.second.wordProbabilities) {
            wordProbabilities.push_back(wordEntry);
        }

        sort(wordProbabilities.begin(), wordProbabilities.end(),
            [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second > b.second;
            });

        int topN = 10;
        for (int i = 0; i < topN && i < wordProbabilities.size(); ++i) {
            cout << "  " << wordProbabilities[i].first << ": " << wordProbabilities[i].second << endl;
        }

        cout << "-------------------------" << endl;
    }
}

void TrainModel::loadModel(const string& filename) {
    ifstream inFile(filename, ios::binary);
    if (!inFile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for reading." << endl;
        return;
    }

    genreModels.clear();
    string genre;
    while (inFile) {
        getline(inFile, genre, '\0');
        if (genre.empty()) continue;

        GenreModel model;
        inFile.read(reinterpret_cast<char*>(&model.priorProbability), sizeof(model.priorProbability));
        inFile.read(reinterpret_cast<char*>(&model.totalWordsInGenre), sizeof(model.totalWordsInGenre));

        while (inFile) {
            string word;
            getline(inFile, word, '\0');
            if (word.empty()) break;

            double wordProb;
            inFile.read(reinterpret_cast<char*>(&wordProb), sizeof(wordProb));
            model.wordProbabilities[word] = wordProb;
        }

        genreModels[genre] = model;
    }

    inFile.close();
    displayModel();
}
