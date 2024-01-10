

from typing import Any
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.ensemble import GradientBoostingClassifier as gbc

class GradientBoostingClassifier(AbstractClassificationModel):
    def __init__(self, **params:dict[str, Any]) -> None:
        super().__init__(**params)

    def _instantiate_model(self, **params:dict[str, Any]) -> gbc:
        gbc_model = gbc(**params)

        return gbc_model

    def _get_params(self, deep:bool = True) -> dict:
        return self._model.get_params(deep=deep)
