import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.v2 import RGB

from instances.models import Classifier


def get_dataset(name, batch_size):
    """
    input: str -> name of the dataset
    return: DataLoader, DataLoader -> train, test splits as torch datasets (keys "image" and "label")
    """
    if name == "letter_recognition":
        n_classes = 26
        return *_letter_recognition(batch_size), n_classes

    elif name == "beans":
        n_classes = 3
        return *_beans(batch_size), n_classes

    elif name == "brain_tumor":
        n_classes = 4
        return *_brain_tumor(batch_size), n_classes

    elif name == "cifar":
        n_classes = 20
        return *_cifar(batch_size), n_classes

    elif name == "cats_and_dogs":
        n_classes = 2
        return *_cats_and_dogs(batch_size), n_classes

    else:
        raise NotImplementedError(name)


def _letter_recognition(batch_size, validation_percent=0.20):
    dataset = load_dataset("pittawat/letter_recognition")
    dataset = dataset.with_format("torch")
    train = dataset["train"]
    test = dataset["test"]

    # create validation split
    validation_size = int(len(train) * validation_percent)
    train_size = len(train) - validation_size
    train_split, validation_split = random_split(train, [train_size, validation_size])

    # create data loaders
    loader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(validation_split, batch_size=batch_size)
    loader_test = DataLoader(test, batch_size=batch_size)

    return loader_train, loader_validation, loader_test


def _beans(batch_size):
    dataset = load_dataset("AI-Lab-Makerere/beans")
    dataset = dataset.rename_column("labels", "label")
    dataset = dataset.with_format("torch")

    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]

    # create data loaders
    loader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(validation, batch_size=batch_size)
    loader_test = DataLoader(test, batch_size=batch_size)

    return loader_train, loader_validation, loader_test


def _brain_tumor(batch_size, validation_percent=0.20):
    dataset = load_dataset("benschill/brain-tumor-collection", trust_remote_code=True)

    # apply preprocessing
    image_pipeline = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    label_pipeline = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8)),
    ])

    def pre_processing(examples):
        examples["image"] = [image_pipeline(image) for image in examples["image"]]
        examples["label"] = [label_pipeline(label) for label in examples["label"]]
        return examples

    dataset.set_transform(pre_processing)
    train = dataset["train"]
    test = dataset["test"]

    # create validation split
    validation_size = int(len(train) * validation_percent)
    train_size = len(train) - validation_size
    train_split, validation_split = random_split(train, [train_size, validation_size])

    # create data loaders
    loader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(validation_split, batch_size=batch_size)
    loader_test = DataLoader(test, batch_size=batch_size)

    return loader_train, loader_validation, loader_test


def _cifar(batch_size, validation_percent=0.20):
    dataset = load_dataset("uoft-cs/cifar100")

    dataset = dataset.rename_column("img", "image")
    dataset = dataset.remove_columns("fine_label")
    dataset = dataset.rename_column("coarse_label", "label")

    dataset = dataset.with_format("torch")

    train = dataset["train"]
    test = dataset["test"]

    # create validation split
    validation_size = int(len(train) * validation_percent)
    train_size = len(train) - validation_size
    train_split, validation_split = random_split(train, [train_size, validation_size])

    # create data loaders
    loader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(validation_split, batch_size=batch_size)
    loader_test = DataLoader(test, batch_size=batch_size)

    return loader_train, loader_validation, loader_test


def _cats_and_dogs(batch_size, validation_percent=0.15, test_percent=0.20):
    dataset = load_dataset("microsoft/cats_vs_dogs")
    dataset = dataset.rename_column("labels", "label")

    # apply preprocessing
    image_pipeline = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        RGB(),
        transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),
    ])

    label_pipeline = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8)),
    ])

    def pre_processing(examples):
        examples["image"] = [image_pipeline(image) for image in examples["image"]]
        examples["label"] = [label_pipeline(label) for label in examples["label"]]
        return examples

    dataset.set_transform(pre_processing)
    train = dataset["train"]

    # create validation and test splits
    validation_size = int(len(train) * validation_percent)
    test_size = int(len(train) * test_percent)
    train_size = len(train) - validation_size - test_size
    train_split, validation_split, test_split = random_split(train, [train_size, validation_size, test_size])

    # create data loaders
    loader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(validation_split, batch_size=batch_size)
    loader_test = DataLoader(test_split, batch_size=batch_size)

    return loader_train, loader_validation, loader_test


def _is_instance_valid(dataset, backbone, model_dir):
    datasets = [
        "letter_recognition",
        "beans",
        "brain_tumor",
        "cifar",
        "cats_and_dogs",
    ]
    backbones = [
        "resnet18",
        "resnet34",
        "resnet50",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "maxvit_t",
        "resnet101",
        # "resnet150",
    ]

    assert dataset in datasets, f"{dataset} is not a valid option. Valid options: {datasets}"
    assert backbone in backbones, f"{backbone} is not valid option. Valid options: {backbones}"

    file_name = f"{backbone}_{dataset}.pt"
    backbone_path = os.path.join(model_dir, file_name)
    assert os.path.exists(backbone_path), (f"{backbone_path} does not exist. "
                                           f"Please, ensure that the fine tuned model is available"
                                           f" at the selected directory: {model_dir}")


def get_instance(dataset, backbone,
                 model_dir=os.path.join("instances", "finetuned_models"),
                 batch_size=32):
    _is_instance_valid(dataset, backbone, model_dir)

    model = torch.load(os.path.join(model_dir, f"{backbone}_{dataset}.pt"),
                       map_location="cpu",
                       weights_only=False)
    model.eval()

    train, validation, test, n_classes = get_dataset(dataset, batch_size)

    return {
        "model": model,
        "data": {
            "train": train,
            "validation": validation,
            "test": test,
            "n_classes": n_classes,
            "batch_size": batch_size,
        }
    }

