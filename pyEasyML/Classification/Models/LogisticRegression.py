

import numpy as np
from os.path import exists
from typing import Any
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.linear_model import LogisticRegression as lr


class LogisticRegression(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> lr:
        lr_model = lr(**params)

        return lr_model

    def _get_params(self) -> dict:
        return self._model.get_params()
