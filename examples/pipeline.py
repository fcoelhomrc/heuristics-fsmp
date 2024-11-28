import os
import sys
import toml
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaheuristics import brkga, problems
from instances import wrappers, models


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

    # get image features from model
    all_features = []
    all_labels = []
    for batch in train:
        image, label = batch["image"], batch["label"]
        features = model.extract_features(image)

        all_features.append(features.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    X_features = np.vstack(all_features)  # (total_samples, n_features_model)
    y = np.concatenate(all_labels)  # (total_samples,)

    end_time_feature_extraction = time.time()
    feature_extraction_time = end_time_feature_extraction - start_time
    print(f"""Time spent on feature extraction: {feature_extraction_time:.2f} seconds 
          (dataset: {config["dataset"]["description"]}, model: {config["model"]["description"]})""")

    # define the feature selection problem
    problem = problems.FeatureSelectionProblem(X_features, y, config["optimization"]["fitness_function"])

    # run brkga
    start_time_brkga = time.time()

    config["brkga"]["eliminate_duplicates"] = (problems.MyElementwiseDuplicateElimination()
                                            if config["brkga"]["eliminate_duplicates"] else None)

    res = brkga.run_algorithm(
        problem=problem,
        algorithm_params=config["brkga"],
        optimization_params=config["optimization"]
    )

    end_time_brkga = time.time()
    brkga_time = end_time_brkga - start_time_brkga
    print(f"Time spent on BRKGA: {brkga_time:.2f} seconds")

    # brkga output
    best_solution_fitness = res.F
    best_solution = np.array(res.X)
    selected_features = np.where(best_solution > 0.5)[0] # threshold for binary

    print("Best solution fitness:", best_solution_fitness)
    print("Number of selected features:", len(selected_features))
    print("Selected features:", selected_features)

    # TODO: check performance using the new subset of features


if __name__ == "__main__":
    main()
