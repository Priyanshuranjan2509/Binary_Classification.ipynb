# Binary Classification Example

This repository contains a simple binary classification project built
around scikit-learn. It provides:

* Data generation utilities (`src/binary_classification/data.py`)
* A training/evaluation script (`src/binary_classification/model.py`)
* A small test suite (`tests/test_model.py`)
* Requirements listed in `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 

## Usage

Train a model from the CLI:

```bash
python -m src.binary_classification.model --output mymodel.joblib
```

Run tests with `pytest`:

```bash
pip install pytest
pytest
```

You can also create a notebook under `notebooks/` to exercise the code interactively.
