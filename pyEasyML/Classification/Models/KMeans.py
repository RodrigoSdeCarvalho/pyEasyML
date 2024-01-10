import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

import numpy as np
import pandas as pd
from Configs.Config import Config
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.cluster import KMeans as km
from typing import Any


class KMeans(AbstractClassificationModel):
    def __init__(self) -> None:
        super().__init__()

    def _instantiate_model(self, **params:dict[str, Any]) -> km:
        km_model = km(**params)

        return km_model

    def _get_params(self) -> dict:
        return self._model.get_params()

    def fit(self, X_train:pd.DataFrame, Y_unsed: Any = None) -> bool:
        try:
            print("Treinando modelo...")
            self._model.fit(X_train)
            self._save_model()
            print('Modelo treinado.')

            return True
    
        except Exception as e:
            print(e)
            return False
