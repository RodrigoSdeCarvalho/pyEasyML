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
