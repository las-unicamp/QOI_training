from typing import Any, Type
from abc import ABC, abstractmethod
from torch import nn
from torchvision.models._api import Weights


class FineTuneModelStrategy(ABC):
    """
    Declare operations common to all implementations of fine tunings.
    This is important since each model has a different fully connected layers
    and number of features.
    """

    @abstractmethod
    def change_num_classes(self, model: Any, num_classes: int) -> Any:
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def change_top_layers(self, model: Any, new_layers: Type[nn.Module]) -> Any:
        raise NotImplementedError("You should implement this!")


class ModelFactory(ABC):
    """
    Factory to separate model creation from its use.
    """

    @abstractmethod
    def get_model(self):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def get_weights(self) -> Weights:
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def get_strategy(self) -> FineTuneModelStrategy:
        raise NotImplementedError("You should implement this!")
