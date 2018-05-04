#!/bin/bash

# Exit on error
set -o errexit

execute_time=10000000

# Run all files in order
# Notebook 0 - Processing Data
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 0.process-data.ipynb

# Notebook 1 - Machine Learning Application
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 1.train-classifier.ipynb

# Notebook 2 - Apply Classifier to Cell Line Data
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 2.apply-classifier.ipynb

# Step 3 - Visualize Coefficients
Rscript --vanilla 3.visualize_coefficients.R

