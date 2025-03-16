#ifndef WORKER_HPP
#define WORKER_HPP

#include <string>
#include <queue>

void workerFunction(int workerId, std::queue<std::string>& workerQueue);

#endif // WORKER_HPP
