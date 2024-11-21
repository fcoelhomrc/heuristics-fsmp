# Introduction

Repository for final project in the course "Heuristics and Metaheuristics" (FEUP, 2024-2025)

The goal is to apply heuristics to tasks within image classification.

There are two components:

- **Feature selection:** find redundant features in the latent space induced by the CNN-based feature extractor
- **Model pruning:** find redundant parameters in a trained model

# Instances 

Our instances are pairs (dataset, model) where we train the model on some image classification task.
The objective functions are defined in terms of some set of metrics computed over the test split. 

Some interesting datasets:
- [Letter Recognition](https://huggingface.co/datasets/pittawat/letter_recognition)
- [Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans)
- [Brain Tumors](https://huggingface.co/datasets/benschill/brain-tumor-collection)
- [Cifar100](https://huggingface.co/datasets/uoft-cs/cifar100)
- [Cats v.s. Dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)

Some backbone models:

| Family    | Model             | Features | Params |
|-----------|-------------------|----------|--------|
| ResNet    | ResNet18          | 512      | 11.7M  |
|           | ResNet34          | 512      | 21.8M  |
|           | ResNet50          | 2048     | 25.6M  |
|           | ResNet101         | 2048     | 44.5M  |
|           | ResNet152         | 2048     | 60.2M  |
| MobileNet | MobileNetV3 Small | 576      | 2.5M   |
|           | MobileNetV3 Large | 960      | 5.5M   |
| MaxViT    | MaxViT T          | 512      | 30.9M  |

# References

- [Knapsack Pruning with Inner Distillation](https://arxiv.org/pdf/2002.08258)
