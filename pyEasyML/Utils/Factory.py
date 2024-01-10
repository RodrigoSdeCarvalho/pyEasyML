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
from Utils.AbstractModel import AbstractModel


class Factory:
    _models:dict[str, AbstractModel] = {}

    @classmethod
    def create(cls, model_name:str, **params:dict[str, Any]) -> AbstractModel:
        if model_name not in cls._models:
            print(f"Modelo {model_name} não encontrado.")
            raise KeyError(f"Modelo {model_name} não encontrado.")

        model = cls._models[model_name]

        return model(**params)

    @classmethod
    def is_model_registered(cls, model_name:str) -> bool:
        return model_name in cls._models
