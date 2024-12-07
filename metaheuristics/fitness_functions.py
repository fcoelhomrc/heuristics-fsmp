import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator


def negative_log_likelihood(y_true:np.ndarray, y_pred_probs:np.ndarray) -> float:
    # extract the probabilities for the true class labels
    row_indices = np.arange(len(y_true))
    column_indices = y_true
    correct_class_probs = y_pred_probs[row_indices, column_indices]

    # negative log-likelihood for the correct class
    return -np.mean(np.log(correct_class_probs))

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
    X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.3, random_state=42)

    if not classifier:
        # classifier = RandomForestClassifier(random_state=42)
        classifier = LogisticRegression(max_iter=200, random_state=42)
        classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    if metric == "accuracy":
        return -accuracy_score(y_test, y_pred)  # pymoo minimizes, so return negative accuracy
    elif metric == "negative_log_likelihood":
        return negative_log_likelihood(y_test, y_pred_proba)
    else:
        return None
