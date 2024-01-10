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
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.ensemble import GradientBoostingClassifier as gbc

class GradientBoostingClassifier(AbstractClassificationModel):
    def __init__(self, **params:dict[str, Any]) -> None:
        super().__init__()

    def _instantiate_model(self, **params:dict[str, Any]) -> gbc:
        gbc_model = gbc(**params)

        return gbc_model

    def _get_params(self, deep:bool = True) -> dict:
        return self._model.get_params(deep=deep)
