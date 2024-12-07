import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class Evaluator():

    def __init__(self,
                 X_train:pd.DataFrame, y_train:pd.DataFrame,
                 X_test:pd.DataFrame, y_test:pd.DataFrame,
                 selected_features:np.ndarray, metric_params:dict
    ):

        self.X_train_all = X_train
        self.X_train_selected = np.take(X_train, selected_features, axis=1)

        self.X_test_all = X_test
        self.X_test_selected = np.take(X_test, selected_features, axis=1)

        self.y_train = y_train
        self.y_test = y_test

        self.metric_params = metric_params

    def _get_metrics(self, y_true:pd.DataFrame, y_score:pd.DataFrame, y_pred:pd.DataFrame) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.metric_params["average"]),
            "recall": recall_score(y_true, y_pred, average=self.metric_params["average"]),
            "f1": f1_score(y_true, y_pred, average=self.metric_params["average"]),
            "roc_auc": roc_auc_score(y_true, y_score, average=self.metric_params["average"], multi_class=self.metric_params["multi_class"])
        }

    def compare_using_fit(self, model_all:BaseEstimator, model_selected:BaseEstimator) -> dict:

        # all features
        model_all.fit(self.X_train_all, self.y_train)
        y_pred_proba_all = model_all.predict_proba(self.X_test_all)
        y_pred_all = model_all.predict(self.X_test_all)

        # selected features
        model_selected.fit(self.X_train_selected, self.y_train)
        y_pred_proba_selected = model_selected.predict_proba(self.X_test_selected)
        y_pred_selected = model_selected.predict(self.X_test_selected)

        # calculate metrics
        return {
            "all_features": self._get_metrics(self.y_test, y_pred_proba_all, y_pred_all),
            "selected_features": self._get_metrics(self.y_test, y_pred_proba_selected, y_pred_selected)
        }

    def compare_using_grid_search_cv(self,
                                     model_all:BaseEstimator, model_selected:BaseEstimator,
                                     cv_params:dict, grid_params:dict
    ) -> dict:

        cv = RepeatedStratifiedKFold(
            n_splits=cv_params["n_splits"],
            n_repeats=cv_params["n_repeats"],
            random_state=cv_params["random_state"]
        )

        # all features
        grid_search_all = GridSearchCV(
            estimator=model_all,
            param_grid=grid_params,
            cv=cv,
            scoring=cv_params["scoring"],
            refit=cv_params["scoring"],
            error_score=0
        )
        grid_search_all.fit(self.X_train_all, self.y_train)

        y_pred_proba_all = grid_search_all.predict_proba(self.X_test_all)
        y_pred_all = grid_search_all.predict(self.X_test_all)

        # selected features
        grid_search_selected = GridSearchCV(
            estimator=model_selected,
            param_grid=grid_params,
            cv=cv,
            scoring=cv_params["scoring"],
            refit=cv_params["scoring"],
            error_score=0
        )
        grid_search_selected.fit(self.X_train_selected, self.y_train)

        y_pred_proba_selected = grid_search_selected.predict_proba(self.X_test_selected)
        y_pred_selected = grid_search_selected.predict(self.X_test_selected)

        # calculate metrics
        return {
            "all_features": self._get_metrics(self.y_test, y_pred_proba_all, y_pred_all),
            "selected_features": self._get_metrics(self.y_test, y_pred_proba_selected, y_pred_selected),
            "best_models": {
                "all_features": grid_search_all.best_params_,
                "selected_features": grid_search_selected.best_params_
            }
        }
