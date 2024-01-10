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
from sklearn.linear_model import Ridge as ridge
from Regression.Models.AbstractRegressionModel import AbstractRegressionModel


class Ridge(AbstractRegressionModel):
    def __init__(self) -> None:
        super().__init__()

    def _instantiate_model(self, **params:dict[str, Any]) -> ridge:
        ridge_model = ridge(**params)

        return ridge_model
