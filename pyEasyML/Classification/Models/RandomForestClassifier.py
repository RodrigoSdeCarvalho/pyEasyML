

from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.ensemble import RandomForestClassifier as RFC
from typing import Any


class RandomForestClassifier(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> RFC:
        rf_model = RFC(**params)

        return rf_model

    def _get_params(self) -> dict:
        return self._model.get_params()
