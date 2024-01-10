

from typing import Any
from xgboost.sklearn import XGBClassifier as xgb
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel


class XGBClassifier(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> xgb:
        xgb_model = xgb(**params)

        return xgb_model

    def _get_params(self, deep:bool = True) -> dict:
        return self._model.get_params(deep=deep)
