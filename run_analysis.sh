#!/bin/bash

execute_time=100000
# Exit on error
set -o errexit

# Run all files in order
jupyter nbconvert --to=notebook \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 0.process-data.ipynb
