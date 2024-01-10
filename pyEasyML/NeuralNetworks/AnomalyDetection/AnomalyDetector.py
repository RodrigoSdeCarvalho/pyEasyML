

os.environ['KERAS_BACKEND'] = 'theano'

from NeuralNetworks.AbstractANN import AbstractANN
from NeuralNetworks.AnomalyDetection.Factory import Factory
from Configs.Config import Config
from Utils.ColumnsToID import ColumnsToID
from Utils.Threshold import Threshold
from sklearn.metrics import confusion_matrix, classification_report
from typing import Any
import numpy as np
from os.path import exists

class AnomalyDetector:
    def __init__(self, 
                 model_name:str,
                 columns:list[str] = None,
                 **model_params:dict[str, Any]) -> None:
        self._config = Config()

        self._columns = self.__get_columns(columns=columns)

        self._columns_id = self.__get_id(columns=columns)

        self._model = self.__get_model(model_name=model_name, **model_params)

        self._thresholds = self.__get_thresholds(model_name=model_name)

    def __get_columns(self, columns:list[str]) -> list[str]:
        if columns is None:
            columns = self._config.SELECTED_FEATURES

        return columns

    def __get_id(self, columns:list[str]) -> tuple[int]:
        if columns is None:
            columns = self._config.SELECTED_FEATURES

        columns_id = ColumnsToID().convert_columns_to_id(*columns)

        return columns_id

    def __get_model(self, model_name:str, **params:dict[str, Any]) -> AbstractANN:
        model = Factory().create(model_name, columns_id=self._columns_id, **params)
        
        return model

    def __get_thresholds(self, model_name:str) -> list[Threshold]:
        thresholds = []
        columns_id = [ColumnsToID().column_id(column) for column in self._columns]
        for column_id in columns_id:
            thresholds.append(Threshold(model=model_name, column_id=column_id))

        return thresholds

    @property
    def model(self) -> AbstractANN:
        return self._model

    @property
    def thresholds(self) -> list[float]:
        thresholds = []
        for threshold in self._thresholds:
            thresholds.append(threshold.value)
            
        return thresholds

    @thresholds.setter
    def thresholds(self, thresholds:list[float]) -> None:
        for threshold, new_value in zip(self._thresholds, thresholds):
            threshold.value = new_value

    def train(self, X_train:np.ndarray, re_train:bool=False, **fit_params:dict[str, Any]) -> bool:
        model_is_trained = self._verify_if_model_trained()

        if not model_is_trained or re_train:
            self._model.fit(X_train, **fit_params)

            healthy_reconstrution = self._model.predict(X_train)
            
            for column_index in range(len(self._thresholds)):
                self._thresholds[column_index].value = self.__reconstruction_mae_loss(original=X_train[:,column_index], reconstructed=healthy_reconstrution[:,column_index])
        
        return True
        
    def _verify_if_model_trained(self) -> bool:
        columns_id_str = ColumnsToID().convert_columns_to_id(*self._columns)
        target_id_str = '0'

        if exists(self._config.get_trained_models_path() + f'{self._model.__class__.__name__}{columns_id_str}{target_id_str}.h5'):
            return True
        else:
            False

    def __reconstruction_mae_loss(self, original:np.ndarray, reconstructed:np.ndarray) -> float:
        loss = self.__reconstruction_loss(original, reconstructed)
        mae_loss = np.mean(loss)

        return mae_loss

    def __reconstruction_loss(self, original:np.ndarray, reconstructed:np.ndarray) -> np.ndarray:
        loss = np.abs(original - reconstructed)
        
        return loss

    def detect(self, dataset:np.ndarray, minimum_anomalous_vars:int=1, threshold_factor:float=1.0) -> np.ndarray:
        reconstructed = self._model.predict(dataset)

        losses = []
        for var in range(dataset.shape[1]):
            losses.append(self.__reconstruction_loss(original=dataset[:,var], reconstructed=reconstructed[:,var]))

        vectorized_verify_if_is_anomaly = np.vectorize(self.__verify_if_is_anomaly)
        detections = []
        for loss, threshold in zip(losses, self._thresholds):
            detection = vectorized_verify_if_is_anomaly(loss=loss, threshold=threshold*threshold_factor)
            detections.append(detection)

        detections = np.column_stack(detections)
        detections = np.apply_along_axis(self.__verify_minimum_anomalous_vars, axis=1, arr=detections, minimum_anomalous_vars=minimum_anomalous_vars)

        return detections

    def __verify_if_is_anomaly(self, loss:float, threshold:Threshold) -> float:
        if loss <= threshold.value:
            return 0.0
        elif loss > threshold.value:
            return 1.0

    def __verify_minimum_anomalous_vars(self, row:Any, minimum_anomalous_vars:int) -> np.ndarray:
        if np.sum(row) >= minimum_anomalous_vars:
            return 1.0
        else:
            return 0.0

    def evaluate(self, X_test:np.ndarray, Y_test:np.ndarray, detections:np.ndarray) -> np.ndarray:
        cm = confusion_matrix(y_true=Y_test, y_pred=detections)
        report = classification_report(y_true=Y_test, y_pred=detections)
        print(report)

        return cm
