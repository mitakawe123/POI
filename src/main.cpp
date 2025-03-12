#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <omp.h>
#include <condition_variable>
#include "train_model.hpp" 
#include "worker.hpp"  // Include the header file for worker functions
#include "utils.hpp"
 // Include the header file for training model

using namespace std;

const int NUM_WORKERS = 4;  // Number of worker threads
std::vector<std::queue<std::string>> workerQueues(NUM_WORKERS);  // Each worker gets its own queue
std::vector<int> workerEfficiencies(NUM_WORKERS, 1);  // Dummy efficiency (can be updated)

std::mutex mtx;  // Mutex to protect shared resources
std::condition_variable cv;  // Condition variable to notify workers
std::vector<bool> workerReady(NUM_WORKERS, false);  // Track worker readiness
bool allFilesAssigned = false;  // Flag to indicate if all files are assigned

int getLeastLoadedWorker() {
    int leastLoadedWorker = 0;
    size_t minQueueSize = workerQueues[0].size();

    #pragma omp parallel for
    for (int i = 1; i < NUM_WORKERS; ++i) {
        size_t currentQueueSize = workerQueues[i].size();
        size_t weightedQueueSize = currentQueueSize * workerEfficiencies[i];

        #pragma omp critical
        {
            if (weightedQueueSize < minQueueSize) {
                minQueueSize = weightedQueueSize;
                leastLoadedWorker = i;
            }
        }
    }
    return leastLoadedWorker;
}

void managerFunction(const std::vector<std::string>& files) {
    // Step 1: Call the training model once (before worker threads)
    TrainModel trainModel;
    try {
        trainModel.trainNaiveBayes();  // Training the model
        trainModel.saveModel("model.dat");  // Saving the trained model
        cout << "Training model completed and saved successfully." << endl;
    } catch (const std::exception& e) {
        cerr << "Error during training model: " << e.what() << endl;
        return;
    }

    // Step 2: Assign files to workers
    cout << "Total files to process: " << files.size() << endl;

    #pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i) {
        std::string file = files[i];
        int workerId = getLeastLoadedWorker();

        #pragma omp critical
        {
            workerQueues[workerId].push(file);
            workerReady[workerId] = true;  // Mark the worker as ready
        }

        cout << "Manager: Assigned file " << file << " to worker " << workerId << endl;
    }

    // Notify workers that all files are assigned
    {
        std::lock_guard<std::mutex> lock(mtx);
        allFilesAssigned = true;
    }
    cv.notify_all();  // Notify all workers to start processing
}

int main() {
    // Read files from the directory
    std::vector<std::string> files = readFilesInDirectory();
    if (files.empty()) {
        cerr << "Error: No files found in the directory!" << endl;
        return 1;
    }

    cout << "Files successfully loaded. Total files: " << files.size() << endl;

    // Create worker threads
    std::vector<std::thread> workerThreads;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workerThreads.push_back(std::thread(workerFunction, i, std::ref(workerQueues[i])));
        cout << "Started worker thread " << i << endl;
    }

    // Distribute work via the manager
    try {
        managerFunction(files);
    } catch (const std::exception& e) {
        cerr << "Error in manager function: " << e.what() << endl;
        return 1;
    }

    // Wait for all workers to finish processing
    try {
        for (auto& thread : workerThreads) {
            thread.join();
        }
        cout << "All workers finished processing." << endl;
    } catch (const std::exception& e) {
        cerr << "Error while joining worker threads: " << e.what() << endl;
        return 1;
    }

    return 0;
}
