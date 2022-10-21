from torch import nn
from torchvision.models import (
    VGG,
    Inception3,
    EfficientNet,
    ResNet,
    VisionTransformer,
)
from ..interfaces import FineTuneModelStrategy


class FineTuneVGG(FineTuneModelStrategy):
    def change_num_classes(self, model: VGG, num_classes: int = 3) -> VGG:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()


class FineTuneInception(FineTuneModelStrategy):
    def change_num_classes(self, model: Inception3, num_classes: int) -> Inception3:
        # Handle auxiliary net
        in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features, num_classes)
        # Handle primary net
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features, 3)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 3)
        )
        return model


class FineTuneEfficientNetB4(FineTuneModelStrategy):
    def change_num_classes(self, model: EfficientNet, num_classes: int) -> EfficientNet:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()


class FineTuneResNet50(FineTuneModelStrategy):
    def change_num_classes(self, model: ResNet, num_classes: int) -> ResNet:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()


class FineTuneViTB16(FineTuneModelStrategy):
    def change_num_classes(
        self, model: VisionTransformer, num_classes: int
    ) -> VisionTransformer:
        in_features = model.heads.head.in_features
        model.heads = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()
