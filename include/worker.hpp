#ifndef WORKER_HPP
#define WORKER_HPP

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "classifier.hpp"
#include "utils.hpp"

// External declarations for synchronization
extern std::mutex modelMutex;
extern std::condition_variable modelLoadedCV;
extern bool modelLoaded;

// External declarations for worker synchronization
extern std::vector<std::queue<std::string>> workerQueues;
extern std::mutex mtx;
extern std::condition_variable cv;
extern bool allFilesAssigned;
extern TrainModel* sharedModel;

// Function to process the file and classify it using the Classifier
void processFileAndClassify(const std::string& file, int workerId, Classifier& classifier);

// Worker function that waits for files and processes them
void workerFunction(int workerId, std::queue<std::string>& workerQueue);

#endif // WORKER_HPP
