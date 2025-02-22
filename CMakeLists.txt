cmake_minimum_required(VERSION 3.31)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include) 
include_directories(${CMAKE_SOURCE_DIR}/data) 
add_executable(main 
    src/main.cpp 
    src/utils.cpp
    src/worker.cpp
    src/classifier.cpp)
