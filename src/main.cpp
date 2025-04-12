#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <thread>
#include <filesystem>
#include <future>
#include "train_model.hpp"
#include "classifier.hpp"
#include "manager.hpp"
#include "worker.hpp"
#include "utils.hpp"

using namespace std;
namespace fs = filesystem;

int NUM_WORKERS = 10;
vector<queue<string>> workerQueues(NUM_WORKERS);
vector<int> workerEfficiencies(NUM_WORKERS, 1); // Initialize worker efficiencies (default value 1)

// Function to load or train the model
unique_ptr<TrainModel> loadOrTrainModel(const string& modelFilename) {
    auto trainModel = make_unique<TrainModel>();

    try {
        if (fs::exists(modelFilename)) {
            trainModel->loadModel(modelFilename);
            cout << "[DEBUG] Model loaded from file: " << modelFilename << endl;
        } else {
            cout << "[DEBUG] Model not found. Training..." << endl;
            trainModel->trainNaiveBayes();
            trainModel->saveModel(modelFilename);
            cout << "[DEBUG] Model trained and saved." << endl;
        }
    } catch (const exception& e) {
        cerr << "[ERROR] Model loading/training failed: " << e.what() << endl;
        return nullptr;
    }

    return trainModel;
}

// Function to handle worker thread initialization
void startWorkerThreads(int numWorkers, vector<thread>& workerThreads, vector<queue<string>>& workerQueues) {
    for (int i = 0; i < numWorkers; ++i) {
        // Start worker thread and pass queue by reference
        workerThreads.emplace_back(workerFunction, i, ref(workerQueues[i]));
        cout << "[DEBUG] Started worker thread " << i << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        try {
            NUM_WORKERS = stoi(argv[1]);
            cout << "[DEBUG] Using " << NUM_WORKERS << " threads." << endl;
        } catch (...) {
            cerr << "[ERROR] Invalid thread count argument. Using default: 10." << endl;
        }
    } else {
        cout << "[DEBUG] No thread count provided. Using default: 10." << endl;
    }

    cout << "[DEBUG] Reading files from the directory..." << endl;

    // Read files asynchronously
    auto filesFuture = async(launch::async, readFilesInDirectory);
    auto files = filesFuture.get();

    if (files.empty()) {
        cerr << "[ERROR] No files found!" << endl;
        return 1;
    }

    cout << "[DEBUG] Files successfully loaded. Total files: " << files.size() << endl;

    // Load or train the model asynchronously
    string modelFilename = "model.dat";
    auto trainModel = loadOrTrainModel(modelFilename);
    if (!trainModel) return 1;

    // Initialize the classifier using the singleton pattern
    Classifier::initialize(*trainModel); // Only need to initialize once
    Classifier& classifier = Classifier::getInstance(); // Access the initialized singleton
    cout << "[DEBUG] Classifier initialized with trained model." << endl;

    // Initialize Manager (pass workerEfficiencies as the second argument)
    Manager manager(NUM_WORKERS, workerQueues, workerEfficiencies);

    // Start worker threads (but they will wait for tasks from the Manager)
    vector<thread> workerThreads;
    startWorkerThreads(NUM_WORKERS, workerThreads, workerQueues);

    // Distribute tasks to workers using Manager
    manager.distributeTasks(files);

    // Wait for all worker threads to finish (if they finish before main thread ends)
    for (auto& thread : workerThreads) {
        thread.join();
    }

    cout << "[DEBUG] All workers finished processing." << endl;

    return 0;
}
