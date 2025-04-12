#!/bin/bash

# Default number of threads
NUM_THREADS=10
EXECUTABLE_NAME="main"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --threads=*)
        NUM_THREADS="${arg#*=}"
        shift
        ;;
        *)
        # unknown option
        ;;
    esac
done

# Record start time
START_TIME=$(date +%s)

# Remove existing build directory (if it exists)
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

# Create build directory
echo "Creating build directory..."
mkdir build

# Navigate to the build directory
cd build

# Run CMake to generate the build files
echo "Running CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Exiting..."
    exit 1
fi

# Build the project using make
echo "Building project..."
make

if [ $? -ne 0 ]; then
    echo "Build failed. Exiting..."
    exit 1
fi

# Run the program with thread count as argument
echo "Running the program with $NUM_THREADS threads..."
./$EXECUTABLE_NAME $NUM_THREADS

if [ $? -ne 0 ]; then
    echo "Failed to run the executable. Exiting..."
    exit 1
fi

# Record end time
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

# Output the elapsed time
echo "Total execution time: $ELAPSED_TIME seconds."