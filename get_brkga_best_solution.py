import itertools
import time
import logging
import toml
import numpy as np
import torch
from instances import wrappers, models
from metaheuristics import brkga, save


# set up logging
logging.basicConfig(
    filename="./log/experiments.log", level=logging.INFO,
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# config file
with open("config.toml", "r") as f:
    config = toml.load(f)

# specify the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_features(input_dataloader, model) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in input_dataloader:
            image, label = batch["image"].to(device), batch["label"].to(device)

            features = model.extract_features(image).detach()

            all_features.append(features)
            all_labels.append(label)

    x_features = torch.cat(all_features).cpu().numpy()  # (total_samples, n_features_model)
    y = torch.cat(all_labels).cpu().numpy()  # (total_samples,)

    return x_features, y

def main(dataset_name, weights, model_name) -> None:

    # load dataset
    train, validation, test, n_classes = wrappers.get_dataset(name=dataset_name, batch_size=config["dataset"]["batch_size"])

    # load model
    model = models.Classifier(backbone=model_name, n_classes=n_classes, n_hidden=config["model"]["n_hidden"])

    logger.info("dataset: %s, n_classes: %d, n_samples: %d, model: %s, weights: %s",
                dataset_name, n_classes, len(train.dataset)+len(validation.dataset), model_name, weights)

    # get image features (training/validation set)
    start_time = time.time()

    x_features_train, y_train = get_image_features(train, model)
    x_features_valid, y_valid = get_image_features(validation, model)

    x_features = np.vstack((x_features_train, x_features_valid))
    y = np.concatenate((y_train, y_valid))

    feature_extraction_time = time.time() - start_time
    logger.info("Time spent on feature extraction: %.2f seconds", feature_extraction_time)

    ########### brkga
    brkga_params = config["brkga"].copy()

    if brkga_params["mode"] == "percent":
        for key in ["n_elites", "n_offsprings", "n_mutants"]:
            brkga_params[key] = int(brkga_params[key] * x_features.shape[1])

    logger.info("BRKGA parameters: n_elites: %d / n_offsprings: %d / n_mutants: %d / bias: %.2f",
                brkga_params["n_elites"], brkga_params["n_offsprings"], brkga_params["n_mutants"], brkga_params["bias"])

    start_time_brkga = time.time()

    res = brkga.run_algorithm(
        X=x_features, y=y,
        algorithm_params=brkga_params,
        optimization_params=config["optimization"]
    )

    brkga_time = time.time() - start_time_brkga
    logger.info("Time spent on BRKGA: %.2f seconds (n_gen: %d)", brkga_time, config["optimization"]["n_gen"])

    # brkga output
    best_solution = np.array(res.X)
    selected_features = np.where(best_solution > brkga_params["threshold_decoding"])[0] # threshold for binary

    logger.info("Best solution fitness: %.6f (metric: %s)", res.F[0], config["optimization"]["fitness_function"])
    logger.info("Number of selected features: %d out of %d (binary threshold: %s)", len(selected_features), x_features.shape[1], brkga_params["threshold_decoding"])

    ########### save results to file
    config_name = dataset_name+"_"+model_name+"_"+weights

    save.brkga_history_to_file(config["file"]["history"].format(name=config_name), res)
    logger.info("Fitness history saved to '%s'", config["file"]["history"].format(name=config_name))

    save.brkga_best_solution_to_file(config["file"]["best_solution"], config_name, best_solution)
    logger.info("Best solution array saved to '%s' with column name '%s'", config["file"]["best_solution"], config_name)

    results_dict = {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "n_samples": len(train.dataset)+len(validation.dataset),
        "model": model_name,
        "n_features_model": x_features.shape[1],
        "weights": weights,
        "time_elapsed_seconds_feature_extraction": feature_extraction_time,
        "brkga_n_elites": brkga_params["n_elites"],
        "brkga_n_offsprings": brkga_params["n_offsprings"],
        "brkga_n_mutants": brkga_params["n_mutants"],
        "brkga_bias": brkga_params["bias"],
        "time_elapsed_seconds_brkga": brkga_time,
        "stop_criterion_n_gen": config["optimization"]["n_gen"],
        "brkga_fitness_function": config["optimization"]["fitness_function"],
        "brkga_best_solution_fitness": res.F[0],
        "brkga_threshold_decoding": brkga_params["threshold_decoding"],
        "brkga_n_selected_features": len(selected_features),
        "file_with_brkga_history": config["file"]["history"].format(name=config_name),
        "file_with_brkga_best_solution": f"{config["file"]["best_solution"]}, column_name: {config_name}"
    }
    save.experiment_results_to_file(config["file"]["all_results"], results_dict)
    logger.info("Results for '%s' saved to '%s'", config_name, config["file"]["all_results"])

if __name__ == "__main__":

    all_datasets = config["dataset"]["description"]
    all_weights = config["dataset"]["weights"]
    all_models = config["model"]["description"]

    for d, w, m in itertools.product(all_datasets, all_weights, all_models):
        logger.info("-----------------------------------------")
        main(d, w, m)
