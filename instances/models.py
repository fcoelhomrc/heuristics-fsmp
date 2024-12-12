import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):

    def __init__(self,
                 backbone: str,
                 n_classes: int,
                 n_hidden: int = 128,
                 ):
        super().__init__()

        self.n_classes = n_classes
        self.n_features = None
        self.n_hidden = n_hidden

        self.model = None
        self._setup_layers(backbone)

    def _setup_layers(self, backbone):
        self.n_features, preprocess, backbone = self._load_backbone(backbone)
        head = self._load_head()
        self.model = nn.ModuleList([
            preprocess,
            backbone,
            head,
        ])

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def extract_features(self, x):
        for layer in self.model[:-1]:
            x = layer(x)
        return x

    @staticmethod
    def _load_backbone(backbone):

        match backbone.lower():
            case 'resnet18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.resnet18(weights=weights)
                n_features = model.fc.in_features
                model.fc = nn.Identity()
                return n_features, preprocess, model

            case 'resnet34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.resnet34(weights=weights)
                n_features = model.fc.in_features
                model.fc = nn.Identity()
                return n_features, preprocess, model

            case 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.resnet50(weights=weights)
                n_features = model.fc.in_features
                model.fc = nn.Identity()
                return n_features, preprocess, model

            case 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.resnet101(weights=weights)
                n_features = model.fc.in_features
                model.fc = nn.Identity()
                return n_features, preprocess, model

            case 'resnet152':
                weights = models.ResNet152_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.resnet152(weights=weights)
                n_features = model.fc.in_features
                model.fc = nn.Identity()
                return n_features, preprocess, model

            case 'mobilenet_v3_small':
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.mobilenet_v3_small(weights=weights)
                n_features = model.classifier[0].in_features
                model.classifier = nn.Identity()
                return n_features, preprocess, model

            case 'mobilenet_v3_large':
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.mobilenet_v3_large(weights=weights)
                n_features = model.classifier[0].in_features
                model.classifier = nn.Identity()
                return n_features, preprocess, model

            case 'maxvit_t':
                weights = models.MaxVit_T_Weights.IMAGENET1K_V1
                preprocess = weights.transforms()
                model = models.maxvit_t(weights=weights)
                n_features = model.classifier[3].in_features
                for i in [3, 4, 5]:
                    model.classifier[i] = nn.Identity()
                return n_features, preprocess, model

            case _:
                raise NotImplementedError(backbone)

    def _load_head(self):
        head = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_classes),
        )
        return head


