import os
import sys
import time
import logging
import toml
import numpy as np

from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from instances import wrappers, models
from metaheuristics import brkga, evaluation, save


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

def log_comparison_results(results, feature_type):
    for metric in results["all_features"]:
        logger.info("%s (test set): %.6f (all features) / %.6f (%s features)",
                    metric, results["all_features"][metric], results["selected_features"][metric], feature_type)

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
    logger.info("dataset: %s, n_classes: %d, n_samples: %d, model: %s"
                ,config["dataset"]["description"], n_classes, len(train.dataset), config["model"]["description"]
    )

    # get image features (training set)
    start_time = time.time()
    X_features_train, y_train = get_image_features(train, model)

    feature_extraction_time = time.time() - start_time
    logger.info("Time spent on feature extraction: %.2f seconds", feature_extraction_time)

    ########### brkga
    if config["brkga"]["mode"] == "percent":
        for key in ["n_elites", "n_offsprings", "n_mutants"]:
            config["brkga"][key] = int(config["brkga"][key] * X_features_train.shape[1])

    logger.info("BRKGA parameters: n_elites: %d / n_offsprings: %d / n_mutants: %d / bias: %.2f",
                config["brkga"]["n_elites"], config["brkga"]["n_offsprings"], config["brkga"]["n_mutants"], config["brkga"]["bias"])

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

    logger.info("Best solution fitness: %.6f (metric: %s)", res.F[0], config["optimization"]["fitness_function"])
    logger.info("Number of selected features: %d out of %d (binary threshold: %s)", len(selected_features), X_features_train.shape[1], config["brkga"]["threshold_decoding"])

    ########### check performance using the new subset of features
    # get image features (test set)
    X_features_test, y_test = get_image_features(test, model)

    # instantiate classifiers
    clf_all = RandomForestClassifier(random_state=19)
    clf_selected = RandomForestClassifier(random_state=19)
    clf_random_selected = RandomForestClassifier(random_state=19)

    ev = evaluation.Evaluator(X_features_train, y_train, X_features_test, y_test, config["metrics"])

    # all features vs selected features
    results = ev.compare_using_fit(clf_all, clf_selected, selected_features)
    log_comparison_results(results, "selected")

    # select the same number of features randomly
    random_selected_features = np.sort(np.random.choice(X_features_train.shape[1], size=len(selected_features), replace=False))
    common_elements = np.intersect1d(selected_features, random_selected_features)
    logger.info("Randomly selected features have %d (%.2f%%) in common with the selected features", len(common_elements), len(common_elements)*100/len(selected_features))

    # all features vs randomly selected features
    results_random = ev.compare_using_fit(clf_all, clf_random_selected, random_selected_features)
    log_comparison_results(results_random, "randomly selected")

    ########### save results to file
    config_name = config["dataset"]["description"]+"_"+config["model"]["description"]+"_imagenet"

    save.brkga_history_to_file(config["file"]["history"].format(name=config_name), res)
    logger.info("Fitness history saved to '%s'", config["file"]["history"].format(name=config_name))

    save.brkga_best_solution_to_file(config["file"]["best_solution"], config_name, best_solution)
    logger.info("Best solution array saved to '%s' with column name '%s'", config["file"]["best_solution"], config_name)

    save.brkga_best_solution_to_file(config["file"]["best_solution"], config_name+"_random", random_selected_features)
    logger.info("Random array saved to '%s' with column name '%s'", config["file"]["best_solution"], config_name+"_random")

    results_dict = {
        "dataset": config["dataset"]["description"],
        "n_classes": n_classes,
        "n_samples_train": len(train.dataset),
        "model": config["model"]["description"],
        "n_features_model": X_features_train.shape[1],
        "weights": "imagenet",  # TODO
        "time_elapsed_seconds_feature_extraction": feature_extraction_time,
        "brkga_n_elites": config["brkga"]["n_elites"],
        "brkga_n_offsprings": config["brkga"]["n_offsprings"],
        "brkga_n_mutants": config["brkga"]["n_mutants"],
        "brkga_bias": config["brkga"]["bias"],
        "time_elapsed_seconds_brkga": brkga_time,
        "stop_criterion_n_gen": config["optimization"]["n_gen"],
        "brkga_fitness_function": config["optimization"]["fitness_function"],
        "brkga_best_solution_fitness": res.F[0],
        "brkga_threshold_decoding": config["brkga"]["threshold_decoding"],
        "brkga_n_selected_features": len(selected_features),
        "file_with_brkga_history": config["file"]["history"].format(name=config_name),
        "file_with_brkga_best_solution": f"{config["file"]["best_solution"]}, column_name: {config_name}",
        "classifier_for_comparison": "RandomForestClassifier",  # TODO
        "accuracy_all": results["all_features"]["accuracy"],
        "accuracy_selected": results["selected_features"]["accuracy"],
        "accuracy_random": results_random["selected_features"]["accuracy"],
        "f1_all": results["all_features"]["f1"],
        "f1_selected": results["selected_features"]["f1"],
        "f1_random": results_random["selected_features"]["f1"],
        "precision_all": results["all_features"]["precision"],
        "precision_selected": results["selected_features"]["precision"],
        "precision_random": results_random["selected_features"]["precision"],
        "recall_all": results["all_features"]["recall"],
        "recall_selected": results["selected_features"]["recall"],
        "recall_random": results_random["selected_features"]["recall"],
        "roc_auc_all": results["all_features"]["roc_auc"],
        "roc_auc_selected": results["selected_features"]["roc_auc"],
        "roc_auc_random": results_random["selected_features"]["roc_auc"],
    }
    save.experiment_results_to_file(config["file"]["all_results"], results_dict)
    logger.info("Results for '%s' saved to '%s'", config_name, config["file"]["all_results"])

if __name__ == "__main__":
    logger.info("-----------------------------------------")
    main()
