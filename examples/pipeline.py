import os
import sys
import toml
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instances import wrappers, models
from metaheuristics import brkga, evaluation


with open("examples/config.toml", "r") as f:
    config = toml.load(f)


def get_image_features(input_dataloader, model):

    all_features = []
    all_labels = []

    for batch in input_dataloader:
        image, label = batch["image"], batch["label"]
        features = model.extract_features(image)

        all_features.append(features.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    X_features = np.vstack(all_features)  # (total_samples, n_features_model)
    y = np.concatenate(all_labels)  # (total_samples,)

    return X_features, y

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
    X_features_train, y_train = get_image_features(train, model)

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
    X_features_test, y_test = get_image_features(test, model)

    # compare performances - all features vs selected features
    clf_all = RandomForestClassifier() # classifier using all features
    clf_selected = RandomForestClassifier() # classifier using selected features

    ev = evaluation.Evaluator(
        X_features_train, y_train,
        X_features_test, y_test,
        selected_features, config["metrics"]
    )

    results = ev.compare_using_fit(clf_all, clf_selected)

    print("\nTest set")
    print("all_features:", results["all_features"])
    print("selected_features:", results["selected_features"])


if __name__ == "__main__":
    main()
