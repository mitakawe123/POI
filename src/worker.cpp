#include "worker.hpp"
#include <iostream>
#include <fstream>
#include <train_model.hpp> 

using namespace std;

// Function to process the file and classify it using the Classifier
void processFileAndClassify(const string& file, int workerId, Classifier& classifier) {
    string fileContent;

    // Read the file content
    try {
        processFile(file, fileContent);
        cout << "[DEBUG] Worker " << workerId << " read file: " << file << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Worker " << workerId << " reading file " << file << ": " << e.what() << endl;
        return;
    }

    // Classify the file using the classifier
    string predictedGenre = classifier.classifyText(fileContent);

    if (predictedGenre.empty()) {
        cerr << "[ERROR] Worker " << workerId << " failed to classify file " << file << endl;
        return;
    }
    
    cout << "[DEBUG] Worker " << workerId << " classified file " << file << " as " << predictedGenre << endl;

    // Write the result to the report file
    lock_guard<mutex> reportLock(mtx);
    ofstream reportFile("classification_report.txt", ios::app);
    if (reportFile.is_open()) {
        reportFile << "File: " << file << ", Predicted Genre: " << predictedGenre << endl;
        reportFile.close();
        cout << "[DEBUG] Worker " << workerId << " wrote to classification_report.txt" << endl;
    } else {
        cerr << "[ERROR] Worker " << workerId << " couldn't open report file!" << endl;
    }
}

// Worker function that waits for files and processes them
void workerFunction(int workerId, queue<string>& workerQueue) {
    try {
        cout << "[DEBUG] Worker " << workerId << " started." << endl;

        // Wait for the model to load
        {
            unique_lock<mutex> modelLock(modelMutex);
            modelLoadedCV.wait(modelLock, [] { return modelLoaded; });
        }
        cout << "[DEBUG] Worker " << workerId << " model is now ready." << endl;

        Classifier classifier(*sharedModel);

        while (true) {
            string file;

            // Lock to check for available files
            {
                unique_lock<mutex> lock(mtx);
                cv.wait(lock, [&] { 
                    return !workerQueue.empty() || allFilesAssigned; 
                });

                if (workerQueue.empty() && allFilesAssigned) {
                    cout << "[DEBUG] Worker " << workerId << " exiting: No more files." << endl;
                    break;
                }

                file = workerQueue.front();
                workerQueue.pop();
            }

            cout << "[DEBUG] Worker " << workerId << " processing file: " << file << endl;
            processFileAndClassify(file, workerId, classifier);
        }

        cout << "[DEBUG] Worker " << workerId << " finished processing." << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Worker " << workerId << ": " << e.what() << endl;
    }
}

