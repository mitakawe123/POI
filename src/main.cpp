#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <omp.h>
#include <condition_variable>
#include "train_model.hpp"
#include "worker.hpp"
#include "utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

const int NUM_WORKERS = 4;
std::vector<std::queue<std::string>> workerQueues(NUM_WORKERS);
std::vector<int> workerEfficiencies(NUM_WORKERS, 1);

std::mutex mtx;
std::condition_variable cv;
std::vector<bool> workerReady(NUM_WORKERS, false);
bool allFilesAssigned = false;

std::mutex modelMutex;
std::condition_variable modelLoadedCV;
bool modelLoaded = false;
TrainModel* sharedModel = nullptr;

// Function to get the least loaded worker
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

    cout << "[DEBUG] Least loaded worker: " << leastLoadedWorker 
         << " with queue size: " << minQueueSize << endl;
    return leastLoadedWorker;
}

// Manager function that assigns work to workers and loads the model
void managerFunction(const std::vector<std::string>& files) {
    // Load or train the model
    TrainModel trainModel;
    std::string modelFilename = "model.dat"; 

    try {
        std::cout << "[DEBUG] Checking if model exists..." << std::endl;
        if (fs::exists(modelFilename)) {
            trainModel.loadModel(modelFilename);
            std::cout << "[DEBUG] Model loaded from file: " << modelFilename << std::endl;
        } else {
            std::cout << "[DEBUG] Model not found. Training..." << std::endl;
            trainModel.trainNaiveBayes();
            trainModel.saveModel(modelFilename);
            std::cout << "[DEBUG] Model trained and saved." << std::endl;
        }

        // Share the model with workers
        {
            std::lock_guard<std::mutex> lock(modelMutex);
            sharedModel = &trainModel;
            modelLoaded = true;
        }
        modelLoadedCV.notify_all();
        std::cout << "[DEBUG] Notified workers that the model is ready." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Model loading/training failed: " << e.what() << std::endl;
        return;
    }

    std::cout << "[DEBUG] Total files to process: " << files.size() << std::endl;

    // Assign files to workers in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i) {
        std::string file = files[i];
        int workerId = getLeastLoadedWorker();

        {
            std::lock_guard<std::mutex> lock(mtx);
            workerQueues[workerId].push(file);
            std::cout << "[DEBUG] Assigned file " << file << " to worker " << workerId
                      << " (Queue size now: " << workerQueues[workerId].size() << ")" << std::endl;
        }
        cv.notify_all();  // Notify workers immediately
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        allFilesAssigned = true;
    }
    cv.notify_all();

    std::cout << "[DEBUG] Notified all workers that all files are assigned." << std::endl;
}


int main() {
    cout << "[DEBUG] Reading files from the directory..." << endl;
    std::vector<std::string> files = readFilesInDirectory();
    if (files.empty()) {
        cerr << "[ERROR] No files found!" << endl;
        return 1;
    }

    cout << "[DEBUG] Files successfully loaded. Total files: " << files.size() << endl;

    std::vector<std::thread> workerThreads;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workerThreads.push_back(std::thread(workerFunction, i, std::ref(workerQueues[i])));
        cout << "[DEBUG] Started worker thread " << i << endl;
    }

    try {
        managerFunction(files);
    } catch (const std::exception& e) {
        cerr << "[ERROR] Manager function error: " << e.what() << endl;
        return 1;
    }

    try {
        for (auto& thread : workerThreads) {
            thread.join();
        }
        cout << "[DEBUG] All workers finished processing." << endl;
    } catch (const std::exception& e) {
        cerr << "[ERROR] Thread join error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
