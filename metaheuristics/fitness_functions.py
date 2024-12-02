import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator


def execute(
    metric:str,
    selected_features:np.ndarray,
    X:np.ndarray,
    y:np.ndarray,
    classifier:BaseEstimator=None
) -> float:
    """Fitness function for BRKGA. Evaluates model on selected features."""

    if selected_features.shape[0] == 0:
        return float("inf") # return a high penalty fitness if no features are selected

    selected_X = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.5, random_state=42)

    if not classifier:
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    if metric == "accuracy":
        return -accuracy_score(y_test, y_pred)  # pymoo minimizes, so return negative accuracy
    else:
        return None
