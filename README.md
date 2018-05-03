# Training and Evaluating a Multiclass Classifier for Multiple Myeloma

**Gregory Way and Casey Greene 2018**

In the following analysis, we train a multiclass classifier on RNAseq data of patients with Multiple Myeloma.
The classifier is trained to distinguish _KRAS_ and _NRAS_ mutations from wild-type tumors.

## Data

The data was provided by Arun Wiita and Tony Lin (UCSF) as part of the [MMRF CoMMpass Study](https://www.themmrf.org/research-partners/mmrf-data-bank/the-mmrf-commpass-study/).
The data was accessed in a Box link provided by Arun and Tony. Before performing the analysis, the appropriate data must be deposited into the appropriate folders.
The data are not included in this repository.

The analysis expects the following data in the `data/raw/` folder:

1. CoMMpass_train_set.csv 
2. CoMMpass_train_set_labels.csv
3. CoMMpass_test_set.csv
4. CoMMpass_test_set_labels.csv
5. MMCL_RNAseq.csv
6. MMCL_RNAseq_labels.csv
7. gprofiler_results_1002952837509.xlsx

If the data are not deposited in the `data/raw` folder, the analysis will not run.
We will provide a reproducible solution by depositing data in an archived and versioned database at a later date.

## Computational Environment

We use conda (`version 4.5.2`) to manage our computational environment.
All scripts must be run in this environment.
After [installing conda](https://conda.io/docs/user-guide/install/index.html), run the following:

```bash
# Initialize the environment
conda env create --force --file environment.yml

# Activate the environment
conda activate multiple-myeloma-classifier
```

## Analysis Pipeline

The analysis consists of a series of notebooks that are designed to be run in order.
See `run_analysis.sh` for more details.

