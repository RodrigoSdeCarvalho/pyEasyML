

from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.svm import SVC as svc


class SVC(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params) -> svc:
        svc_model = svc(**params)

        return svc_model
