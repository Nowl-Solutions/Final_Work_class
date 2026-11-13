"""Base models for your own ensemble implementations.

GA means Genetic Algorithm here. Each factory returns fresh, unfitted models.
"""

from pygad import GA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def make_classifiers(random_state=42):
    """Models for predicting Iris Species. Scale features before fitting."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", random_state=random_state),
        "naive_bayes": GaussianNB(),
    }


def make_regressors():
    """Models for a numeric target, such as petal length, rather than Species."""
    return {
        "linear_regression": LinearRegression(),
        "knn_regression": KNeighborsRegressor(n_neighbors=5),
        "svm_regression": SVR(kernel="rbf"),
    }


def make_kmeans(random_state=42):
    """Cluster IDs are not Species labels; fit using X only."""
    return KMeans(n_clusters=3, n_init=10, random_state=random_state)


# GA is available via: from base_models import GA
# Supply your own fitness_func and GA settings when creating GA(...).
