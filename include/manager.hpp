#ifndef MANAGER_HPP
#define MANAGER_HPP

#include <queue>
#include <string>
#include <vector>

class Manager {
private:
    int numWorkers;
    std::vector<std::queue<std::string>>& workerQueues;  // Reference to the workers' task queues
    std::vector<int>& workerEfficiencies;  // Efficiency values of the workers

public:
    // Constructor that initializes the manager with workers' queues and efficiencies
    Manager(int numWorkers, std::vector<std::queue<std::string>>& queues, std::vector<int>& efficiencies);

    // Function to find the least loaded worker based on queue size and worker efficiency
    int getLeastLoadedWorker();

    // Function to distribute tasks among workers dynamically based on their load
    void distributeTasks(const std::vector<std::string>& files);
};

#endif // MANAGER_HPP
