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

    std::getline(file, line);  // Skip header (index and title)

    while (std::getline(file, line)) {
        std::wstringstream ss(line);
        std::wstring dummy;

        // Skip the first two columns (index and title)
        std::getline(ss, dummy, L',');  // Skip first column (index)
        std::getline(ss, dummy, L',');  // Skip second column (title)

        // Now get the genre from the third column
        std::wstring genre;
        std::getline(ss, genre, L',');

        // If genre is empty, continue with next line
        if (genre.empty()) continue;

        // Read the summary from the remaining columns (might span multiple lines)
        std::wstring summary;
        bool inSummary = true;
        std::wstring summaryLine;
        std::getline(ss, summaryLine);  // Start reading the summary (fourth column)

        // Add the initial summary line
        summary += summaryLine;

        // Now, read the rest of the summary until we encounter "(less)"
        while (std::getline(file, line)) {
            // If the line contains "(less)", we stop reading the summary
            if (line.find(L"(less)") != std::wstring::npos) {
                break;
            }

            // Skip empty lines
            if (line.empty()) continue;

            // Add the line to the summary
            summary += L" " + line;
        }

        // Store the genre and the summary in the vector
        rows.push_back({std::string(genre.begin(), genre.end()), std::string(summary.begin(), summary.end())});
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

    // Parallelizing the document processing using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < trainingData.size(); ++i) {
        std::string genre = trainingData[i].first;

        // Only process genres that are in predefinedGenres
        if (std::find(predefinedGenres.begin(), predefinedGenres.end(), genre) == predefinedGenres.end()) {
            continue;  // Skip this document if the genre is not in predefinedGenres
        }

        std::string summary = trainingData[i].second;
        std::vector<std::string> words = preprocessText(summary);

        #pragma omp critical
        {
            genreDocumentCounts[genre]++;
            totalDocuments++;

            // Count the word occurrences per genre (critical section)
            for (const std::string& word : words) {
                tempWordCounts[genre][word]++;
            }
        }

        // Debugging: Output progress every 1000 documents processed
        if (i % 1000 == 0) {
            std::cout << "Processed " << i << " documents..." << std::endl;
        }
    }

    // Debugging: Output intermediate information about total documents processed
    std::cout << "Total Documents Processed: " << totalDocuments << std::endl;

    // Building the model for each genre
    #pragma omp parallel for
    for (auto& genreEntry : tempWordCounts) {
        std::string genre = genreEntry.first;
        GenreModel model;
        model.totalWordsInGenre = 0;

        for (const auto& wordEntry : genreEntry.second) {
            model.wordProbabilities[wordEntry.first] = static_cast<double>(wordEntry.second);
            model.totalWordsInGenre += wordEntry.second;
        }

        model.priorProbability = static_cast<double>(genreDocumentCounts[genre]) / totalDocuments;

        // Normalize word probabilities
        for (auto& wordEntry : model.wordProbabilities) {
            wordEntry.second /= model.totalWordsInGenre;
        }

        // Store the final model for the genre
        #pragma omp critical
        {
            genreModels[genre] = model;
        }

        // Debugging: Output progress after processing each genre
        std::cout << "Finished processing genre: " << genre << std::endl;
    }

    displayModel();
}

void TrainModel::addGenreModel(const std::string& genre, GenreModel& genreModel) {
    genreModels[genre] = genreModel;
}

void TrainModel::saveModel(const std::string& filename) {
    string directory = "./models/";  // Relative path don't use absolute path Yo
    string fullPath = directory + filename;

    if (!fs::exists(directory)) {
        if (fs::create_directory(directory)) {
            cout << "Created directory: " << directory << endl;
        } else {
            cerr << "Error: Failed to create directory: " << directory << endl;
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

    std::cout << "Loading model from: " << filename << std::endl;

    while (inFile.peek() != EOF) {
        std::string genre;
        std::getline(inFile, genre, '\0');

        GenreModel genreModel;
        inFile.read(reinterpret_cast<char*>(&genreModel.priorProbability), sizeof(genreModel.priorProbability));
        inFile.read(reinterpret_cast<char*>(&genreModel.totalWordsInGenre), sizeof(genreModel.totalWordsInGenre));

        while (inFile.peek() != EOF) {
            std::string word;
            std::getline(inFile, word, '\0');
            if (word.empty()) break;

            double probability;
            inFile.read(reinterpret_cast<char*>(&probability), sizeof(probability));
            genreModel.wordProbabilities[word] = probability;
        }

        genreModels[genre] = genreModel;
    }

    inFile.close();
    std::cout << "Model loaded successfully." << std::endl;
}
