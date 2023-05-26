import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from typing import Any

# Created models will be imported here.
from NeuralNetworks.AbstractANN import AbstractANN
from NeuralNetworks.Autoencoders.DeepAutoencoder import DeepAutoencoder

# Import of Singleton
from Utils.Singleton import Singleton

class Factory(Singleton):
    def __init__(self) -> None:
        if not super().created:
            self._models:dict[str, AbstractANN] = {
                "DeepAutoencoder": DeepAutoencoder
            }

    def create(self, model_name:str, **params:dict[str, Any]) -> AbstractANN:
        if model_name not in self._models:
            print(f"Modelo {model_name} não encontrado.")
            raise KeyError(f"Modelo {model_name} não encontrado.")

        model = self._models[model_name]

        return model(**params)
