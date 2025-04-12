#include "manager.hpp"
#include <iostream>
#include <omp.h>
#include <queue>
#include <vector>
#include <limits>

using namespace std;

Manager::Manager(int numWorkers, std::vector<std::queue<std::string>>& queues, std::vector<int>& efficiencies)
    : numWorkers(numWorkers), workerQueues(queues), workerEfficiencies(efficiencies) {
    // No need for atomic variables here, we will rely on the efficiency of dynamic scheduling.
}

int Manager::getLeastLoadedWorker() {
    int leastLoadedWorker = 0;
    size_t minQueueSize = std::numeric_limits<size_t>::max();

    // Check for the least loaded worker
    for (int i = 0; i < numWorkers; ++i) {
        // Calculate weighted load for the worker (could be based on queue size and efficiency)
        size_t weightedLoad = workerQueues[i].size() * workerEfficiencies[i]; // Just example logic
        if (weightedLoad < minQueueSize) {
            minQueueSize = weightedLoad;
            leastLoadedWorker = i;
        }
    }

    return leastLoadedWorker;
}

void Manager::distributeTasks(const vector<string>& files) {
    cout << "[DEBUG] Starting task distribution. Total files: " << files.size() << endl;

    // Distribute files to workers
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < files.size(); ++i) {
            int workerId = getLeastLoadedWorker();  // Select the least loaded worker dynamically

            // Push the file into the selected queue
            workerQueues[workerId].push(files[i]);

            // Log the assignment — ensure thread-safe output
            #pragma omp critical
            {
                cout << "[DEBUG] Assigned file \"" << files[i] << "\" to worker " << workerId << endl;
            }
        }
    }

    // After task distribution, add exit signals to each queue
    for (size_t i = 0; i < workerQueues.size(); ++i) {
        workerQueues[i].push("__EXIT__");  // Send exit signal to each worker
        cout << "[DEBUG] Sent exit signal to worker " << i << endl;
    }

    // After parallel region — log queue sizes
    for (size_t i = 0; i < workerQueues.size(); ++i) {
        cout << "[DEBUG] Queue " << i << " size after distribution: " << workerQueues[i].size() << endl;
    }

    cout << "[DEBUG] Task distribution completed." << endl;
}
