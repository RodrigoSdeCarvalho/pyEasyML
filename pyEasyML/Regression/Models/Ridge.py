

from typing import Any
from sklearn.linear_model import Ridge as ridge
from Regression.Models.AbstractRegressionModel import AbstractRegressionModel


class Ridge(AbstractRegressionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> ridge:
        ridge_model = ridge(**params)

        return ridge_model
