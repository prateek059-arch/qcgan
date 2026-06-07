# QC-GAN UNSW-NB15 Replication Pipeline

This repository contains a compact replication pipeline for training and comparing a quantum-classical GAN against a classical GAN baseline on the UNSW-NB15 intrusion detection dataset.

The workflow covers:

- Exploratory data analysis
- Feature cleanup
- Train/validation/test split preparation
- Missing-value imputation
- Feature selection for a 4-qubit GAN
- Classical GAN, clean QC-GAN, and noisy QC-GAN training/comparison

## Files

`eda.py`  
Runs initial dataset inspection and saves summary outputs under `preprocessing_outputs/`.

`feature_drop.py`  
Drops identifier, high-missing, and constant or near-constant columns.

`traintest.py`  
Creates train, validation, and test CSV splits from the cleaned dataset.

`imputation.py`  
Handles numeric and categorical missing values and saves imputed splits.

`feature_select.py`  
Ranks features using Random Forest, Mutual Information, and L1 Logistic Regression. It saves exactly four selected GAN features to:

```text
preprocessing_outputs/models/feature_selection/selected_features.json
preprocessing_outputs/models/feature_selection/top_4_features.json
```

`finalcomp.py`  
Trains and compares:

- Classical GAN
- QC-GAN
- QC-GAN with noise

If `selected_features.json` exists, `finalcomp.py` uses those four selected features. Otherwise it falls back to:

```python
["synack", "ct_state_ttl", "sbytes", "smean"]
```

`requirements.txt`  
Python package dependencies.

## Data Setup

Download the UNSW-NB15 dataset and place these files in the project root:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

If your files are inside a folder such as `UNSW_NB15/`, either move them to the root or update the paths in `eda.py` and `feature_drop.py`.

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Order

Run the scripts in this order:

```powershell
python eda.py
python feature_drop.py
python traintest.py
python imputation.py
python feature_select.py
python finalcomp.py
```

The preprocessing scripts write outputs to `preprocessing_outputs/`.

The final comparison script writes results to:

```text
spie_results/
```

## Notes

The quantum GAN is configured for four features because the model uses four qubits. Run `feature_select.py` before `finalcomp.py` if you want the GANs to train on the selected four-feature subset.

Training time depends heavily on CPU/GPU availability and PennyLane backend performance. Installing `pennylane-lightning` is recommended for faster clean quantum simulation.
