

from typing import Any

# Created models will be imported here.
from NeuralNetworks.AbstractANN import AbstractANN
from NeuralNetworks.Autoencoders.DeepAutoencoder import DeepAutoencoder

# Import of Singleton
from Utils.Singleton import singleton


@singleton
class Factory:
    def __init__(self) -> None:
        self._models: dict[str, AbstractANN] = {
            "DeepAutoencoder": DeepAutoencoder
        }

    def create(self, model_name: str, **params: dict[str, Any]) -> AbstractANN:
        if model_name not in self._models:
            print(f"Modelo {model_name} não encontrado.")
            raise KeyError(f"Modelo {model_name} não encontrado.")

        model = self._models[model_name]

        return model(**params)
