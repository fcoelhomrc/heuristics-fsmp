from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.v2 import RGB

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