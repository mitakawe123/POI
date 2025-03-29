#include "worker.hpp"
#include <iostream>
#include <fstream>
#include <classifier.hpp>  // Include the classifier header
#include <sstream>         // Include the sstream header
#include <utils.hpp>       // Include the utils header

// Worker function that processes tasks from the queue
void workerFunction(int workerId, std::queue<std::string>& workerQueue) {
    try {
        std::cout << "[DEBUG] Worker " << workerId << " started." << std::endl;

        // Get the singleton instance of Classifier (it will handle the model internally)
        Classifier& classifier = Classifier::getInstance();

        while (true) {
            std::string file;

            // If no more files in the queue, exit the loop
            if (workerQueue.empty()) {
                break;
            }

            // Get the file from the queue
            file = workerQueue.front();
            workerQueue.pop();

            std::cout << "[DEBUG] Worker " << workerId << " processing file: " << file << std::endl;

            // Process the file (read content)
            std::string fileContent;
            try {
                processFile(file, fileContent);
                std::cout << "[DEBUG] Worker " << workerId << " read file: " << file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Worker " << workerId << " reading file " << file << ": " << e.what() << std::endl;
                continue;  // Skip this file and continue with the next one
            }

            // Classify the text using the classifier (without knowing the model)
            std::string predictedGenre = classifier.classifyText(fileContent);  // Classifier is responsible for model usage

            if (!predictedGenre.empty()) {
                std::cout << "[DEBUG] Worker " << workerId << " classified file " << file << " as " << predictedGenre << std::endl;

                // Write the result to the report file
                std::ofstream reportFile("classification_report.txt", std::ios::app);
                if (reportFile.is_open()) {
                    reportFile << "File: " << file << ", Predicted Genre: " << predictedGenre << std::endl;
                    reportFile.close();
                    std::cout << "[DEBUG] Worker " << workerId << " wrote to classification_report.txt" << std::endl;
                } else {
                    std::cerr << "[ERROR] Worker " << workerId << " couldn't open report file!" << std::endl;
                }
            } else {
                std::cerr << "[ERROR] Worker " << workerId << " failed to classify file " << file << std::endl;
            }
        }

        cout << "[DEBUG] Worker " << workerId << " finished processing." << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Worker " << workerId << ": " << e.what() << endl;
    }
}
