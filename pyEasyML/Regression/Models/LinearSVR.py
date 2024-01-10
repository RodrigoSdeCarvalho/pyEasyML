from typing import Any
from sklearn.svm import LinearSVR as lsvr
from Regression.Models.AbstractRegressionModel import AbstractRegressionModel


class LinearSVR(AbstractRegressionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params:dict[str, Any]) -> lsvr:
        lsvr_model = lsvr(**params)

        return lsvr_model
