#!/bin/bash

# Clean build directory
rm -rf build
mkdir -p build
cd build

# Configure and build
cmake ..
make -j$(nproc)  # Uses all available CPU cores
