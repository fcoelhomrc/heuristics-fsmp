import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def accuracy(selected_features:np.ndarray, X:np.ndarray, y:np.ndarray) -> float:
    """Fitness function for BRKGA. Evaluates model accuracy on selected features."""

    if selected_features.shape[0] == 0:
        return float("inf") # return a high penalty fitness if no features are selected

    selected_X = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.5, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return -accuracy_score(y_test, y_pred)  # pymoo minimizes, so return negative accuracy
