

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
