

from abc import ABC, abstractmethod
from Configs.Config import Config
from os.path import exists
from typing import Any
from keras.models import Sequential, load_model

class AbstractANN(ABC):
    def __init__(self, columns_id = 0, target_id = 0) -> None:
        self._config = Config()
        self._columns_id = columns_id
        self._target_id = target_id
        self._model:Sequential = None # will be instantiated in _instantiate_model(), by the child class.

    @property
    def columns_id(self) -> int:
        return self._columns_id

    @columns_id.setter
    def columns_id(self, columns_id:int) -> None:
        self._columns_id = columns_id

    @property
    def target_id(self) -> int:
        return self._target_id

    @target_id.setter
    def target_id(self, target_id:int) -> None:
        self._target_id = target_id

    @property
    def model(self) -> Sequential:
        return self._model

    @abstractmethod
    def _instantiate_model(self) -> Sequential:
        pass

    @abstractmethod
    def fit(self) -> bool:
        pass
    
    @abstractmethod
    def predict(self) -> Any:
        pass

    def _save_model(self) -> None:
        model_path = os.path.join(self._config.get_trained_models_path(), self.__class__.__name__ + str(self.columns_id) + str(self._target_id) + '.h5')
        self._model.save(model_path)

    def _load_model(self) -> Sequential:
        model_path = os.path.join(self._config.get_trained_models_path(), self.__class__.__name__ + str(self.columns_id) + str(self._target_id) + '.h5')

        if not exists(model_path):
            return None
        else:
            loaded_model = load_model(model_path)
            return loaded_model

    @abstractmethod
    def evaluate(self) -> Any:
        pass
