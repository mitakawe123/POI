cmake_minimum_required(VERSION 3.31)

# Add the project name and version
project(POI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Enable debugging symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Include directories for headers and data
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/data)

# Define the executable
add_executable(main 
    src/main.cpp 
    src/utils.cpp
    src/worker.cpp
    src/classifier.cpp
    src/train_model.cpp
    src/genre_model.cpp)
