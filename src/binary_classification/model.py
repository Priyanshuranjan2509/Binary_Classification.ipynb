"""Model training and evaluation utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from .data import generate_dataset


def train(output_path: str = "model.joblib", **data_kwargs):
    """Generate data, train a logistic regression model, and save it.

    Returns the trained model and accuracy on a hold-out set.
    """
    X, y = generate_dataset(**data_kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump(clf, output_path)
    return clf, acc


def load_model(path: str):
    """Load a saved model from disk."""
    return joblib.load(path)


def evaluate(model, X, y):
    """Evaluate a model on provided data and return accuracy."""
    preds = model.predict(X)
    return accuracy_score(y, preds)


if __name__ == "__main__":
    # simple CLI when run as script
    import argparse
    parser = argparse.ArgumentParser(description="Train a binary classification model")
    parser.add_argument("--output", default="model.joblib", help="where to save trained model")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--features", type=int, default=20)
    args = parser.parse_args()
    model, accuracy = train(
        output_path=args.output,
        n_samples=args.samples,
        n_features=args.features,
    )
    print(f"Trained model with accuracy: {accuracy:.4f}")
