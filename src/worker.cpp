#include "worker.hpp"
#include <iostream>
#include <fstream>
#include <classifier.hpp> 
#include <sstream>        
#include <utils.hpp>      
#include <thread>   
#include <chrono>   

using namespace std;

// Worker function that processes tasks from the queue
void workerFunction(int workerId, std::queue<std::string>& workerQueue) {
    try {
        std::cout << "[DEBUG] Worker " << workerId << " started." << std::endl;

        Classifier& classifier = Classifier::getInstance();

        while (true) {
            if (workerQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Wait a bit before retrying
                continue;
            }

            std::string file = workerQueue.front();
            workerQueue.pop();

            // Check for shutdown signal
            if (file == "__EXIT__") {
                std::cout << "[DEBUG] Worker " << workerId << " received shutdown signal." << std::endl;
                break;
            }

            std::cout << "[DEBUG] Worker " << workerId << " processing file: " << file << std::endl;

            // Read and classify file
            std::string fileContent;
            try {
                processFile(file, fileContent);
                std::cout << "[DEBUG] Worker " << workerId << " read file: " << file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Worker " << workerId << " reading file " << file << ": " << e.what() << std::endl;
                continue;
            }

            std::string predictedGenre = classifier.classifyText(fileContent);

            if (!predictedGenre.empty()) {
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

        std::cout << "[DEBUG] Worker " << workerId << " finished processing." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Worker " << workerId << ": " << e.what() << std::endl;
    }
}
