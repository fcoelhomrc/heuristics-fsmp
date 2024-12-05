import os
import sys
import toml
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instances import wrappers, models
from metaheuristics import brkga


with open("examples/config.toml", "r") as f:
    config = toml.load(f)


def main():

    start_time = time.time()

    # load dataset
    train, validation, test, n_classes = wrappers.get_dataset(
        name=config["dataset"]["description"],
        batch_size=config["dataset"]["batch_size"]
    )

    # load model
    model = models.Classifier(
        backbone=config["model"]["description"],
        n_classes=n_classes,
        n_hidden=config["model"]["n_hidden"]
    )

    # get image features (training set)
    all_features_train = []
    all_labels_train = []
    for batch in train:
        image, label = batch["image"], batch["label"]
        features = model.extract_features(image)

        all_features_train.append(features.detach().cpu().numpy())
        all_labels_train.append(label.detach().cpu().numpy())

    X_features_train = np.vstack(all_features_train)  # (total_samples, n_features_model)
    y_train = np.concatenate(all_labels_train)  # (total_samples,)

    end_time_feature_extraction = time.time()
    feature_extraction_time = end_time_feature_extraction - start_time
    print(f"Time spent on feature extraction: {feature_extraction_time:.2f} seconds (dataset: {config["dataset"]["description"]}, model: {config["model"]["description"]})")

    # run brkga
    start_time_brkga = time.time()

    res = brkga.run_algorithm(
        X=X_features_train,
        y=y_train,
        algorithm_params=config["brkga"],
        optimization_params=config["optimization"]
    )

    end_time_brkga = time.time()
    brkga_time = end_time_brkga - start_time_brkga
    print(f"Time spent on BRKGA: {brkga_time:.2f} seconds (n_gen: {config["optimization"]["n_gen"]})")

    # brkga output
    best_solution_fitness = res.F
    best_solution = np.array(res.X)
    selected_features = np.where(best_solution > 0.5)[0] # threshold for binary

    print(f"\nBest solution fitness: {best_solution_fitness[0]} (metric: {config["optimization"]["fitness_function"]})")
    print("Number of selected features:", len(selected_features))
    print("Total number of features:", X_features_train.shape[1])

    # algorithm history
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    print("Number of function evaluations:", n_evals)
    print("Fitness history:", opt)

    ########### check performance using the new subset of features

    # get image features (test set)
    all_features_test = []
    all_labels_test = []
    for batch in test:
        image, label = batch["image"], batch["label"]
        features = model.extract_features(image)

        all_features_test.append(features.detach().cpu().numpy())
        all_labels_test.append(label.detach().cpu().numpy())

    X_features_test = np.vstack(all_features_test)  # (total_samples, n_features_model)
    y_test = np.concatenate(all_labels_test)  # (total_samples,)

    # overview - number of features
    print(f"\nAll features: X_train shape: {X_features_train.shape}, X_test shape: {X_features_test.shape}")
    X_features_train_opt = np.take(X_features_train, selected_features, axis=1)
    X_features_test_opt = np.take(X_features_test, selected_features, axis=1)
    print(f"Selected features: X_train shape: {X_features_train_opt.shape}, X_test shape: {X_features_test_opt.shape}")
 
    # classifier using all features
    clf_full = RandomForestClassifier()
    clf_full.fit(X_features_train, y_train)
    y_pred_proba_full = clf_full.predict_proba(X_features_test)
    y_pred_full = clf_full.predict(X_features_test)

    # classifier using selected features
    clf_opt = RandomForestClassifier()
    clf_opt.fit(X_features_train_opt, y_train)
    y_pred_proba_opt = clf_opt.predict_proba(X_features_test_opt)
    y_pred_opt = clf_opt.predict(X_features_test_opt)

    # results - test set
    print("\nTest set")
    print(f"- Accuracy:\nAll features: {accuracy_score(y_test, y_pred_full)}\nSelected features: {accuracy_score(y_test, y_pred_opt)}")
    print(f"- Precision:\nAll features: {precision_score(y_test, y_pred_full, average="macro")}\nSelected features: {precision_score(y_test, y_pred_opt, average="macro")}")
    print(f"- Recall:\nAll features: {recall_score(y_test, y_pred_full, average="macro")}\nSelected features: {recall_score(y_test, y_pred_opt, average="macro")}")
    print(f"- F1 Score:\nAll features: {f1_score(y_test, y_pred_full, average="macro")}\nSelected features: {f1_score(y_test, y_pred_opt, average="macro")}")
    print(f"- ROC AUC:\nAll features: {roc_auc_score(y_test, y_pred_proba_full, average="macro", multi_class="ovr")}\nSelected features: {roc_auc_score(y_test, y_pred_proba_opt, average="macro", multi_class="ovr")}")


if __name__ == "__main__":
    main()
