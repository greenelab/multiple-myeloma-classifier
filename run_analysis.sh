#!/bin/bash

# Exit on error
set -o errexit

# Export to .py files
jupyter nbconvert --to=python --template=util/rm_magic.tpl --FilesWriter.build_directory=scripts *.ipynb

# Run all files in order
python scripts/0.get_mad_genes.py

