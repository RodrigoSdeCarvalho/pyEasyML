

from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class KNeighborsClassifier(AbstractClassificationModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _instantiate_model(self, **params) -> KNN:
        knn_model = KNN(**params)

        return knn_model
    
    def _get_params(self) -> dict:
        return self._model.get_params()
