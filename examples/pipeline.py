import os
import sys
import time
import logging
import toml
import numpy as np

from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instances import wrappers, models
from metaheuristics import brkga, evaluation


# set up logging
logging.basicConfig(
    filename="./log/debug.log", level=logging.INFO,
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# config file
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
    start_time = time.time()
    X_features_train, y_train = get_image_features(train, model)

    feature_extraction_time = time.time() - start_time
    logger.info("Time spent on feature extraction: %.2f seconds (dataset: %s, model: %s)", feature_extraction_time, config["dataset"]["description"], config["model"]["description"])

    # run brkga
    start_time_brkga = time.time()
    res = brkga.run_algorithm(
        X=X_features_train,
        y=y_train,
        algorithm_params=config["brkga"],
        optimization_params=config["optimization"]
    )

    brkga_time = time.time() - start_time_brkga
    logger.info("Time spent on BRKGA: %.2f seconds (n_gen: %d)", brkga_time, config["optimization"]["n_gen"])

    # brkga output
    best_solution = np.array(res.X)
    selected_features = np.where(best_solution > config["brkga"]["threshold_decoding"])[0] # threshold for binary

    logger.info("Best solution fitness: %.4f (metric: %s)", res.F[0], config["optimization"]["fitness_function"])
    logger.info("Number of selected features: %d out of %d", len(selected_features), X_features_train.shape[1])
    # logger.info("Number of function evaluations: %s", np.array([e.evaluator.n_eval for e in res.history]))
    # logger.info("Fitness history: %s", np.array([e.opt[0].F for e in res.history]))

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
    # results = ev.compare_using_grid_search_cv(clf_all, clf_selected, config["cross_validation"], config["random_forest"])

    for metric in results["all_features"]:
        logger.info("%s (test set): %.4f (all features) / %.4f (selected features)", metric, results["all_features"][metric], results["selected_features"][metric])
    # print("best models:", results["best_models"])


if __name__ == "__main__":
    logger.info("-----------------------------------------")
    main()
