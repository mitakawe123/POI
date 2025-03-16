#include "manager.hpp"
#include <iostream>
#include <omp.h>
#include <queue>
#include <vector>
#include <limits>

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

void Manager::distributeTasks(const std::vector<std::string>& files) {
    std::cout << "[DEBUG] Starting task distribution. Total files: " << files.size() << std::endl;

    // Instead of locking mechanisms, just select the least loaded worker dynamically
    #pragma omp parallel
    {
        // Each thread works independently and assigns tasks to the least loaded worker
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < files.size(); ++i) {
            int workerId = getLeastLoadedWorker();  // Select the least loaded worker dynamically
            // Assign the task to the least loaded worker
            workerQueues[workerId].push(files[i]);
        }
    }

    std::cout << "[DEBUG] Task distribution completed." << std::endl;
}
