"""Load local Iris data and evaluate individual classifiers."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from base_models import make_classifiers

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"


def load_iris_split(test_size=0.2, random_state=42):
    """Return unscaled train/test arrays and the label encoder.

    Id is excluded. Fit preprocessing on training data inside each learner or
    validation fold, then transform its held-out data with the same scaler.
    """
    frame = pd.read_csv(DATA_PATH)
    X = frame[FEATURES].to_numpy(dtype=float)
    X_train, X_test, labels_train, labels_test = train_test_split(
        X,
        frame["Species"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=frame["Species"],
    )
    encoder = LabelEncoder().fit(labels_train)
    return (
        X_train,
        X_test,
        encoder.transform(labels_train),
        encoder.transform(labels_test),
        encoder,
    )


def main():
    X_train, X_test, y_train, y_test, encoder = load_iris_split()
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {len(FEATURES)}")
    print(f"Classes: {dict(enumerate(encoder.classes_.tolist()))}")
    for name, model in make_classifiers().items():
        learner = Pipeline([("scale", StandardScaler()), ("model", model)])
        learner.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, learner.predict(X_test))
        print(f"{name:22s} accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
