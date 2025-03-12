#ifndef WORKER_HPP
#define WORKER_HPP

#include <string>
#include <queue>

// Declare the functions (do not define them here)
void processFile(const std::string& file);
void workerFunction(int workerId, std::queue<std::string>& workerQueue);


#endif // WORKER_HPP
