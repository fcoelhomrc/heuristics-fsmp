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
- [Brain Tumors](https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification)
- [Cifar100](https://huggingface.co/datasets/uoft-cs/cifar100)
- [Cats v.s. Dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
- [Chest X-Rays](https://huggingface.co/datasets/AiresPucrs/chest-xray)

Some base models:
- [MobileViT - XXS](https://huggingface.co/apple/mobilevit-xx-small) (1.3M parameters)
- [MobileViT - XS](https://huggingface.co/apple/mobilevit-x-small) (2.3M parameters)
- [MobileViT - S](https://huggingface.co/apple/mobilevit-small) (5.6M parameters)

- [ResNet 18](https://huggingface.co/microsoft/resnet-18) (11.7M parameters)
- [ResNet 26](https://huggingface.co/microsoft/resnet-26) (16M parameters)
- [ResNet 34](https://huggingface.co/microsoft/resnet-34) (21.8M parameters)
- [ResNet 50](https://huggingface.co/microsoft/resnet-50) (25.6M parameters)
- [ResNet 101](https://huggingface.co/microsoft/resnet-101) (44.5M parameters)

# References

- [Knapsack Pruning with Inner Distillation](https://arxiv.org/pdf/2002.08258)
