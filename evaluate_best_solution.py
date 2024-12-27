import toml
import torch
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from instances import wrappers, models
from metaheuristics import evaluation, save


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

    # get the best solution from file
    config_name = dataset_name+"_"+model_name+"_"+weights

    with open(config["file"]["best_solution"]) as file:
        headers = file.readline().strip().split(",")
        if config_name not in headers:
            return
        usecols_index = headers.index(config_name)

    best_solution = np.genfromtxt(config["file"]["best_solution"], delimiter=",", usecols=usecols_index, skip_header=1)
    selected_features = np.where(best_solution > config["brkga"]["threshold_decoding"])[0]

    # load dataset
    train, validation, test, n_classes = wrappers.get_dataset(name=dataset_name, batch_size=config["dataset"]["batch_size"])

    # load model
    model = models.Classifier(backbone=model_name, n_classes=n_classes, n_hidden=config["model"]["n_hidden"])

    # get image features (training/validation set)
    x_features_t, y_t = get_image_features(train, model)
    x_features_v, y_v = get_image_features(validation, model)

    x_features_train = np.vstack((x_features_t, x_features_v))
    y_train = np.concatenate((y_t, y_v))

    # get image features (test set)
    x_features_test, y_test = get_image_features(test, model)

    # instantiate classifiers
    clf = "RandomForestClassifier"
    clf_all = RandomForestClassifier(random_state=19)
    clf_selected = RandomForestClassifier(random_state=19)
    clf_random_selected = RandomForestClassifier(random_state=19)

    ev = evaluation.Evaluator(x_features_train, y_train, x_features_test, y_test, config["metrics"])
    all_results = []

    # all features vs selected features
    results = ev.compare_using_fit(clf_all, clf_selected, selected_features)

    all_results.append({
        "dataset": dataset_name,
        "model": model_name,
        "weights": weights,
        "n_features_model": x_features_train.shape[1],
        "feature selection": "all features",
        "selected_features": None,
        "common_elements_random_choice": None,
        "classifier_for_comparison": clf,
        "accuracy": results["all_features"]["accuracy"],
        "f1": results["all_features"]["f1"],
        "precision": results["all_features"]["precision"],
        "recall": results["all_features"]["recall"],
        "roc_auc": results["all_features"]["roc_auc"],
    })
    all_results.append({
        "dataset": dataset_name,
        "model": model_name,
        "weights": weights,
        "n_features_model": x_features_train.shape[1],
        "feature selection": "brkga features",
        "selected_features": len(selected_features),
        "common_elements_random_choice": None,
        "classifier_for_comparison": clf,
        "accuracy": results["selected_features"]["accuracy"],
        "f1": results["selected_features"]["f1"],
        "precision": results["selected_features"]["precision"],
        "recall": results["selected_features"]["recall"],
        "roc_auc": results["selected_features"]["roc_auc"],
    })

    # select the same number of features randomly - do it N times
    random_evaluations = 10
    for _ in range(random_evaluations):
        random_selected_features = np.sort(np.random.choice(x_features_train.shape[1], size=len(selected_features), replace=False))
        common_elements = np.intersect1d(selected_features, random_selected_features)

        results_random = ev.fit_predict_random(clf_random_selected, random_selected_features)
        all_results.append({
            "dataset": dataset_name,
            "model": model_name,
            "weights": weights,
            "n_features_model": x_features_train.shape[1],
            "feature selection": "random features",
            "selected_features": len(selected_features),
            "common_elements_random_choice": len(common_elements),
            "classifier_for_comparison": clf,
            "accuracy": results_random["random_features"]["accuracy"],
            "f1": results_random["random_features"]["f1"],
            "precision": results_random["random_features"]["precision"],
            "recall": results_random["random_features"]["recall"],
            "roc_auc": results_random["random_features"]["roc_auc"],
        })

    save.experiment_results_to_file(config["file"]["evaluation"], all_results)

if __name__ == "__main__":

    all_datasets = config["dataset"]["description"]
    all_weights = config["dataset"]["weights"]
    all_models = config["model"]["description"]

    for d, w, m in itertools.product(all_datasets, all_weights, all_models):
        main(d, w, m)
