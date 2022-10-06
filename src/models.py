from abc import ABC, abstractmethod
from typing import Any, Type
from torch import nn
from torchvision import models
from torchvision.models._api import Weights
from torchvision.models import (
    VGG,
    Inception3,
    EfficientNet,
    ResNet,
    VisionTransformer,
    VGG11_BN_Weights,
    Inception_V3_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)


class FineTuneModelStrategy(ABC):
    @abstractmethod
    def change_num_classes(self, model: Any, num_classes: int) -> Any:
        pass

    def change_top_layers(self, model: Any, new_layers: Type[nn.Module]) -> Any:
        pass


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
        raise NotImplementedError()


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


class ModelFactory(ABC):
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_weights(self) -> Weights:
        pass

    @abstractmethod
    def get_strategy(self) -> FineTuneModelStrategy:
        pass


class VGGModel(ModelFactory):
    def get_model(self):
        return models.vgg11_bn

    def get_weights(self):
        return VGG11_BN_Weights.IMAGENET1K_V1

    def get_strategy(self):
        return FineTuneVGG()


class InceptionModel(ModelFactory):
    def get_model(self):
        return models.inception_v3

    def get_weights(self):
        return Inception_V3_Weights.IMAGENET1K_V1

    def get_strategy(self):
        return FineTuneInception()


class EfficientNetB4Model(ModelFactory):
    def get_model(self):
        return models.efficientnet_b4

    def get_weights(self):
        return EfficientNet_B4_Weights.IMAGENET1K_V1

    def get_strategy(self):
        return FineTuneEfficientNetB4()


class ResNet50Model(ModelFactory):
    def get_model(self):
        return models.resnet50

    def get_weights(self):
        return ResNet50_Weights.IMAGENET1K_V1

    def get_strategy(self):
        return FineTuneResNet50()


class ViTB16Model(ModelFactory):
    def get_model(self):
        return models.vit_b_16

    def get_weights(self):
        return ViT_B_16_Weights.IMAGENET1K_V1

    def get_strategy(self):
        return FineTuneViTB16()


def map_model(model_name):
    factories = {
        "vgg": VGGModel(),
        "inception": InceptionModel(),
        "efficient": EfficientNetB4Model(),
        "resnet": ResNet50Model(),
        "vit": ViTB16Model(),
    }

    if model_name not in factories:
        raise ValueError(f"Only {factories.keys()} are available.")

    return factories[model_name]


def fine_tune(model_name):
    factory = map_model(model_name)

    model_fn = factory.get_model()
    weights = factory.get_weights()
    strategy = factory.get_strategy()

    model = model_fn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model = strategy.change_num_classes(model, num_classes=3)
    return model


def main():
    model_name = "vit"

    model = fine_tune(model_name)
    print(model)


if __name__ == "__main__":
    main()
