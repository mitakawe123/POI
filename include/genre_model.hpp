#ifndef GENRE_MODEL_HPP
#define GENRE_MODEL_HPP

#include <unordered_map>
#include <string>

class GenreModel {
public:
    std::string genre;  // The genre name (e.g., "thriller")
    double priorProbability;  // Prior probability for the genre
    int totalWordsInGenre;  // Total number of words in the genre
    std::unordered_map<std::string, double> wordProbabilities;  // Probability of each word in the genre

    // Default constructor (declaration only)
    GenreModel();
    
    // Constructor
    GenreModel(std::string genre, double priorProb, int totalWords);
};

#endif
