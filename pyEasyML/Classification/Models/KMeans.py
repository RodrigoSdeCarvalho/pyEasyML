

import numpy as np
import pandas as pd
from Configs.Config import Config
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.cluster import KMeans as km
from typing import Any


class KMeans(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
