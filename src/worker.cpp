#include "worker.hpp"
#include <iostream>
#include <string>
#include <queue>
#include <exception>
#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include <condition_variable>
#include "utils.hpp"
#include "classifier.hpp"

using namespace std;

extern std::vector<std::queue<std::string>> workerQueues;
extern std::vector<bool> workerReady;  // Track worker readiness
extern std::mutex mtx;
extern std::condition_variable cv;
extern bool allFilesAssigned;  // Flag to indicate if all files are assigned

void processFile(const std::string& file, string& fileContent) {
    ifstream fileStream(file);
    if (!fileStream) {
        throw runtime_error("Error opening file: " + file);
    }

    string line;
    while (getline(fileStream, line)) {
        fileContent += line + " ";  // Append lines with a space
    }
}

void workerFunction(int workerId, std::queue<std::string>& workerQueue) {
    try {
        // Load the model (only once per worker)
        Classify classifier;
        classifier.loadModel("trained_model.dat");  // Adjust model file name as needed

        while (true) {
            std::string file;

            // Wait until all files are assigned to workers
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [] { return allFilesAssigned; });  // Wait until files are assigned
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                // Process the files if available
                if (!workerQueue.empty()) {
                    file = workerQueue.front();
                    workerQueue.pop();
                } else {
                    // If there are no files left, break the loop
                    break;
                }
            }

            if (!file.empty()) {
                cout << "Worker " << workerId << " is processing: " << file << endl;
                string fileContent;
                processFile(file, fileContent);

                // Classify the file using Naive Bayes
                string predictedGenre = classifier.classifyWithNaiveBayes(fileContent);
                cout << "Worker " << workerId << " predicted genre: " << predictedGenre << endl;

                // Write the result to a report file
                ofstream reportFile("classification_report.txt", ios::app);
                if (reportFile.is_open()) {
                    reportFile << "File: " << file << ", Predicted Genre: " << predictedGenre << endl;
                    reportFile.close();
                    cout << "Worker " << workerId << " successfully wrote to the report file." << endl;
                } else {
                    cout << "Worker " << workerId << " failed to open classification report file." << endl;
                }
            }
        }
    } catch (const std::exception& e) {
        cout << "Error in worker " << workerId << ": " << e.what() << endl;
    }
}
