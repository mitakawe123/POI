#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <condition_variable>
#include "train_model.hpp"
#include "worker.hpp"
#include "utils.hpp"
#include <filesystem>

using namespace std;
namespace fs = filesystem;

const int NUM_WORKERS = 4;
vector<queue<string>> workerQueues(NUM_WORKERS);
vector<int> workerEfficiencies(NUM_WORKERS, 1);

mutex mtx;
condition_variable cv;
vector<bool> workerReady(NUM_WORKERS, false);
bool allFilesAssigned = false;

mutex modelMutex;
condition_variable modelLoadedCV;
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
void managerFunction(const vector<string>& files) {
    // Load or train the model
    TrainModel trainModel;
    string modelFilename = "model.dat"; 

    try {
        cout << "[DEBUG] Checking if model exists..." << endl;
        if (fs::exists(modelFilename)) {
            trainModel.loadModel(modelFilename);
            cout << "[DEBUG] Model loaded from file: " << modelFilename << endl;
        } else {
            cout << "[DEBUG] Model not found. Training..." << endl;
            trainModel.trainNaiveBayes();
            trainModel.saveModel(modelFilename);
            cout << "[DEBUG] Model trained and saved." << endl;
        }

        // Share the model with workers
        {
            lock_guard<mutex> lock(modelMutex);
            sharedModel = &trainModel;
            modelLoaded = true;
        }
        modelLoadedCV.notify_all();
        cout << "[DEBUG] Notified workers that the model is ready." << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Model loading/training failed: " << e.what() << endl;
        return;
    }

    cout << "[DEBUG] Total files to process: " << files.size() << endl;

    // Assign files to workers in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < files.size(); ++i) {
        string file = files[i];
        int workerId = getLeastLoadedWorker();

        {
            lock_guard<mutex> lock(mtx);
            workerQueues[workerId].push(file);
            cout << "[DEBUG] Assigned file " << file << " to worker " << workerId
                      << " (Queue size now: " << workerQueues[workerId].size() << ")" << endl;
        }
        cv.notify_all();  // Notify workers immediately
    }

    {
        lock_guard<mutex> lock(mtx);
        allFilesAssigned = true;
    }
    cv.notify_all();

    cout << "[DEBUG] Notified all workers that all files are assigned." << endl;
}


int main() {
    cout << "[DEBUG] Reading files from the directory..." << endl;
    vector<string> files = readFilesInDirectory();
    if (files.empty()) {
        cerr << "[ERROR] No files found!" << endl;
        return 1;
    }

    cout << "[DEBUG] Files successfully loaded. Total files: " << files.size() << endl;

    vector<thread> workerThreads;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workerThreads.push_back(thread(workerFunction, i, ref(workerQueues[i])));
        cout << "[DEBUG] Started worker thread " << i << endl;
    }

    try {
        managerFunction(files);
    } catch (const exception& e) {
        cerr << "[ERROR] Manager function error: " << e.what() << endl;
        return 1;
    }

    try {
        for (auto& thread : workerThreads) {
            thread.join();
        }
        cout << "[DEBUG] All workers finished processing." << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Thread join error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
