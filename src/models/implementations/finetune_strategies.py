import torch
from torch import nn
from torchvision.models import (
    VGG,
    Inception3,
    EfficientNet,
    ResNet,
    VisionTransformer,
)
from ..interfaces import FineTuneModelStrategy


def copy_weights_channelwise(layer, weights, num_copies):
    with torch.no_grad():
        for i in range(num_copies):
            index_start = i * 3
            index_end = (i + 1) * 3
            layer.weight[:, index_start:index_end] = weights
    return layer


class FineTuneVGG(FineTuneModelStrategy):
    def change_num_classes(self, model: VGG, num_classes: int = 3) -> VGG:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()

    def change_first_conv_layer(self, model, num_input_images):
        first_layer = model.features[0]
        weight = first_layer.weight.clone()
        first_layer = nn.Conv2d(
            3 * num_input_images, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        first_layer = copy_weights_channelwise(first_layer, weight, num_input_images)
        model.features[0] = first_layer
        return model


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

    def change_first_conv_layer(self, model, num_input_images):
        first_layer = model.Conv2d_1a_3x3.conv
        weight = first_layer.weight.clone()
        first_layer = nn.Conv2d(
            3 * num_input_images, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )
        first_layer = copy_weights_channelwise(first_layer, weight, num_input_images)
        model.Conv2d_1a_3x3.conv = first_layer
        return model


class FineTuneEfficientNetB4(FineTuneModelStrategy):
    def change_num_classes(self, model: EfficientNet, num_classes: int) -> EfficientNet:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()

    def change_first_conv_layer(self, model, num_input_images):
        first_layer = model.features[0][0]
        weight = first_layer.weight.clone()
        first_layer = nn.Conv2d(
            3 * num_input_images,
            48,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        first_layer = copy_weights_channelwise(first_layer, weight, num_input_images)
        model.features[0][0] = first_layer
        return model


class FineTuneResNet50(FineTuneModelStrategy):
    def change_num_classes(self, model: ResNet, num_classes: int) -> ResNet:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()

    def change_first_conv_layer(self, model, num_input_images):
        first_layer = model.conv1
        weight = first_layer.weight.clone()
        first_layer = nn.Conv2d(
            3 * num_input_images,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        first_layer = copy_weights_channelwise(first_layer, weight, num_input_images)
        model.conv1 = first_layer
        return model


class FineTuneViTB16(FineTuneModelStrategy):
    def change_num_classes(
        self, model: VisionTransformer, num_classes: int
    ) -> VisionTransformer:
        in_features = model.heads.head.in_features
        model.heads = nn.Linear(in_features, num_classes)
        return model

    def change_top_layers(self, model, new_layers):
        raise NotImplementedError()

    def change_first_conv_layer(self, model, num_input_images):
        first_layer = model.conv_proj
        weight = first_layer.weight.clone()
        first_layer = nn.Conv2d(
            3 * num_input_images, 768, kernel_size=(16, 16), stride=(16, 16)
        )
        first_layer = copy_weights_channelwise(first_layer, weight, num_input_images)
        model.conv_proj = first_layer
        return model
