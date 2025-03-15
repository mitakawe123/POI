#include "worker.hpp"
#include <iostream>
#include <fstream>
#include <train_model.hpp>  // Ensure this is correctly included

// Function to process the file and classify it using the Classifier
void processFileAndClassify(const std::string& file, int workerId, Classifier& classifier) {
    std::string fileContent;

    // Read the file content
    try {
        processFile(file, fileContent);
        std::cout << "[DEBUG] Worker " << workerId << " read file: " << file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Worker " << workerId << " reading file " << file << ": " << e.what() << std::endl;
        return;
    }

    // Classify the file using the classifier
    std::string predictedGenre = classifier.classifyText(fileContent);

    if (!predictedGenre.empty()) {
        std::cout << "[DEBUG] Worker " << workerId << " classified file " << file << " as " << predictedGenre << std::endl;

        // Write the result to the report file
        std::lock_guard<std::mutex> reportLock(mtx);
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

// Worker function that waits for files and processes them
void workerFunction(int workerId, std::queue<std::string>& workerQueue) {
    try {
        std::cout << "[DEBUG] Worker " << workerId << " started." << std::endl;

        // Wait for the model to load
        {
            std::unique_lock<std::mutex> modelLock(modelMutex);
            modelLoadedCV.wait(modelLock, [] { return modelLoaded; });
        }
        std::cout << "[DEBUG] Worker " << workerId << " model is now ready." << std::endl;

        Classifier classifier(*sharedModel);

        while (true) {
            std::string file;

            // Lock to check for available files
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !workerQueue.empty() || allFilesAssigned; });

                if (workerQueue.empty() && allFilesAssigned) {
                    std::cout << "[DEBUG] Worker " << workerId << " exiting: No more files." << std::endl;
                    break;
                }

                file = workerQueue.front();
                workerQueue.pop();
            }

            std::cout << "[DEBUG] Worker " << workerId << " processing file: " << file << std::endl;
            processFileAndClassify(file, workerId, classifier);
        }

        std::cout << "[DEBUG] Worker " << workerId << " finished processing." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Worker " << workerId << ": " << e.what() << std::endl;
    }
}

