"""Data generation utilities for binary classification."""

from sklearn.datasets import make_classification
import pandas as pd


def generate_dataset(n_samples: int = 1000, n_features: int = 20, random_state: int = 42):
    """Generate a synthetic binary classification dataset.

    Returns a tuple `(X, y)` where `X` is a pandas DataFrame and `y` is a pandas Series.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    labels = pd.Series(y, name="target")
    return df, labels
