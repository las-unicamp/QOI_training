from torchvision import models
from torchvision.models import (
    VGG11_BN_Weights,
    Inception_V3_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)
from .finetune_strategies import (
    FineTuneEfficientNetB4,
    FineTuneInception,
    FineTuneResNet50,
    FineTuneVGG,
    FineTuneViTB16,
)
from ..interfaces import ModelFactory


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
