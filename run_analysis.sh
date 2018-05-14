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

# Currently commented out - see https://github.com/greenelab/multiple-myeloma-classifier/issues/3
# for more details
# Notebook 3 - Visualize Coefficients
# jupyter nbconvert --to=html \
#        --FilesWriter.build_directory=html \
#        --ExecutePreprocessor.kernel_name=R \
#        --execute 3.visualize-coefficients.ipynb

Rscript scriptes/3.visualize-coefficients.r

