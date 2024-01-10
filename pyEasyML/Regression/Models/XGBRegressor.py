

from typing import Any
from xgboost import XGBRegressor as xgb
from Regression.Models.AbstractRegressionModel import AbstractRegressionModel


class XGBRegressor(AbstractRegressionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> xgb:
        xgb_model = xgb(**params)

        return xgb_model

    def _get_params(self, deep:bool = True) -> dict:
        return self._model.get_params(deep=deep)
