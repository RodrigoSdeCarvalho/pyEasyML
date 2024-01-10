

from typing import Any
from NeuralNetworks.Layers.Layers import Layers

class DeepAutoencoderLayers(Layers):
    def __init__(self) -> None:
        super().__init__()

    def add(self, **params:dict[str, Any]) -> None:
        super().add("Dense", **params)

    def reverse(self) -> list[int]:
        units = [layer.units for layer in self._layers]
        units.reverse()

        return units
