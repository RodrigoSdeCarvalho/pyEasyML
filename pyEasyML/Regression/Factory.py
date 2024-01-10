

from typing import Any

from Regression.Models.LinearSVR import LinearSVR
from Regression.Models.Ridge import Ridge
from Regression.Models.XGBRegressor import XGBRegressor
from Utils.Factory import Factory as BaseFactory
from Utils.AbstractModel import AbstractModel


class Factory(BaseFactory):
    _models:dict[str, AbstractModel] = {
        "LinearSVR": LinearSVR,
        "Ridge": Ridge,
        "XGBRegressor": XGBRegressor
    }
