from enum import Enum, auto
from src.models.implementations.models import (
    VGGModel,
    InceptionModel,
    EfficientNetB4Model,
    ResNet50Model,
    ViTB16Model,
)


class AvailableModels(Enum):
    VGG = auto()
    INCEPTION = auto()
    EFFICIENT = auto()
    RESNET = auto()
    VIT = auto()


def _factory_model(model_name: AvailableModels):
    """
    Function to retrieve the model, its weights and the strategy to finetune it.

    Args:
        model_name (str): name of the model

    Return:
        (tuple): model_fn, weights, strategy
    """

    factories = {
        model_name.VGG: VGGModel(),
        model_name.INCEPTION: InceptionModel(),
        model_name.EFFICIENT: EfficientNetB4Model(),
        model_name.RESNET: ResNet50Model(),
        model_name.VIT: ViTB16Model(),
    }

    if model_name not in factories:
        raise ValueError(f"Only {factories.keys()} are available.")

    factory = factories[model_name]

    model_fn = factory.get_model()
    weights = factory.get_weights()
    fine_tune_strategy = factory.get_strategy()

    return model_fn, weights, fine_tune_strategy


def get_model(model_name: AvailableModels, num_classes: int):
    """
    Function to get the model and its preprocessing steps.

    Args:
        model_name (str): name of the model
        num_classes (str): number of final output classes/values

    Return:
        (tuple): model instance, preprocessing
    """
    # load model, pre-trained weights and strategy to finetune the model
    model_fn, weights, fine_tune_strategy = _factory_model(model_name)

    # initialize the model
    model = model_fn(weights=weights)

    # freezes the network
    for param in model.parameters():
        param.requires_grad = False

    # perform feature extracing (this makes the last layer trainable)
    model = fine_tune_strategy.change_num_classes(model, num_classes=num_classes)
    # model = fine_tune_strategy.change_top_layers(model, new_layers=None)

    # one can also redefine the entire classifier/regressor with:
    # fine_tune_strategy.change_top_layers(model, new_layers)

    # get the preprocessing expected by the selected model
    transform = weights.transforms()

    return model, transform


def test():
    model_name = AvailableModels.VGG

    model_fn, weights, fine_tune_strategy = _factory_model(model_name)

    model = model_fn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model = fine_tune_strategy.change_num_classes(model, num_classes=3)
    print(model)

    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name, param.shape)


if __name__ == "__main__":
    test()
