#!/bin/bash

# Set the name of the executable (optional)
EXECUTABLE_NAME="main"

# Check if the build directory exists, if not, create it
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to the build directory
cd build

# Run CMake to generate the build files
echo "Running CMake..."
cmake ..

# Check if the cmake command was successful
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Exiting..."
    exit 1
fi

# Build the project using make
echo "Building project..."
make

# Check if the make command was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting..."
    exit 1
fi

# Run the executable
echo "Running the program..."
./$EXECUTABLE_NAME

# Check if the executable ran successfully
if [ $? -ne 0 ]; then
    echo "Failed to run the executable. Exiting..."
    exit 1
fi
