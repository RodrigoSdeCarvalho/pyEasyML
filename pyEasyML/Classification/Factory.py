

from typing import Any

# Created models will be imported here.
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel as AbstractModel

from Classification.Models.SVC import SVC
from Classification.Models.XGBClassifier import XGBClassifier
from Classification.Models.KMeans import KMeans
from Classification.Models.LogisticRegression import LogisticRegression
from Classification.Models.GradientBoostingClassifier import GradientBoostingClassifier

from Utils.Factory import Factory as BaseFactory


class Factory(BaseFactory):
    _models:dict[str, AbstractModel] = {
        "SVC": SVC,
        "XGBClassifier": XGBClassifier,
        "KMeans": KMeans,
        "LogisticRegression": LogisticRegression,
        "GradientBoostingClassifier": GradientBoostingClassifier
    }