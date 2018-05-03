#!/bin/bash

# Exit on error
set -o errexit

execute_time=10000000

# Run all files in order
conda activate multiple-myeloma-classifier

# Notebook 1 - Processing Data
jupyter nbconvert --to=notebook \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 0.process-data.ipynb

# Notebook 2 - Machine Learning Application
jupyter nbconvert --to=notebook \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 1.train-classifier.ipynb

# Notebook 3 - Apply Classifier to Cell Line Data
jupyter nbconvert --to=notebook \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 2.apply-classifier.ipynb

# Visualize Coefficients
Rscript --vanilla visualize_coefficients.R

