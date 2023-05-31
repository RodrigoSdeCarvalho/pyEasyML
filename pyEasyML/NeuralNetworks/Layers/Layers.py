import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from typing import Any, Dict, List, Type, Union
from keras.engine.base_layer import Layer
from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D

class Layers:
    def __init__(self) -> None:
        self._factory:dict[str, Layer] = {
            "Input": Input,
            "Dense": Dense,
            "Dropout": Dropout,
            "Conv2D": Conv2D,
            "MaxPooling2D": MaxPooling2D,
            "UpSampling2D": UpSampling2D,
            "Conv1D": Conv1D,
            "MaxPooling1D": MaxPooling1D,
            "UpSampling1D": UpSampling1D,
        }

        self._layers:list[Layer] = []

    def add(self, name:str, **params:dict[str, Any]) -> None:
        new_layer = self._factory[name](**params)
        self._layers.append(new_layer)

    def insert_at(self, index:int, name:str, **params:dict[str, Any]) -> None:
        new_layer = self._factory[name](**params)
        self._layers.insert(index, new_layer)

    def __iter__(self) -> iter:
        return iter(self._layers)

    def __getitem__(self, index: Union[int, slice]) -> Union['Layer', List['Layer']]:
        return self._layers[index]

    def __setitem__(self, index: Union[int, slice], value: Union['Layer', List['Layer']]) -> None:
        if isinstance(index, int):
            self._layers[index] = value
        elif isinstance(index, slice):
            self._layers[index] = value
        else:
            raise TypeError("Invalid index type. Expected int or slice.")

    def __len__(self) -> int:
        return len(self._layers)


if __name__ == '__main__':
    # TEST HERE
    pass
