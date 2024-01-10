import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.ensemble import RandomForestClassifier as RFC
from typing import Any


class RandomForestClassifier(AbstractClassificationModel):
    def __init__(self) -> None:
        super().__init__()

    def _instantiate_model(self, **params:dict[str, Any]) -> RFC:
        rf_model = RFC(**params)

        return rf_model

    def _get_params(self) -> dict:
        return self._model.get_params()
