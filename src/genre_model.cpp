#include "genre_model.hpp"

// Default constructor definition
GenreModel::GenreModel() : priorProbability(0.0), totalWordsInGenre(0) {}

// Constructor definition
GenreModel::GenreModel(std::string genre, double priorProb, int totalWords)
    : genre(genre), priorProbability(priorProb), totalWordsInGenre(totalWords) {}
