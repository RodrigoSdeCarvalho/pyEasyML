import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from abc import ABC, abstractmethod
import pandas as pd
import pickle
from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config
from os.path import exists
from typing import Any
import numpy as np


class AbstractModel(ABC):
    """
    Classe abstrata que define o modelo de regressão a ser utilizado.
    O nome da classe deve ser o mesmo nome da classe do modelo a ser utilizado.
    A interface provavelmente será mantida para modelos da biblioteca sklearn.
    Ao importar o modelo da sklearn, recomenda-se que usa um alias, para evitar conflitos de nomes na IDE.
    """
    def __init__(self) -> None:
        self._config = Config()
        self._columns_to_id = ColumnsToID()

        self._columns = None # Will both be set in the constructor of the Classifier class.
        self._target = None

        self._model = self._instantiate_model()

    @property
    def model(self) -> Any:
        return self._model

    @property
    def columns(self) -> list[str]:
        return self._columns

    @columns.setter
    def columns(self, columns:list[str]) -> None:
        self._columns = columns

    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, target:str) -> None:
        self._target = target

    @abstractmethod
    def _instantiate_model(self, **kwargs) -> Any:
        pass

    def _get_params(self) -> dict:
        return self._model.get_params()

    def fit(self, X_train:pd.DataFrame, Y_train:pd.DataFrame) -> bool:
        try:
            print("Treinando modelo...")
            self._model.fit(X_train, Y_train)
            self._save_model()
            print('Modelo treinado.')

            return True
        except Exception as e:
            print(e)
            return False

    def predict(self, dataset:np.ndarray) -> np.ndarray:
        self._model = self._load_model()

        return self._model.predict(dataset)

    def predict_proba(self, dataset:np.ndarray) -> np.ndarray:
        self._model = self._load_model()

        return self._model.predict_proba(dataset)

    def _save_model(self) -> None:
        print("Salvando modelo...")
        model_type = self._model.__class__.__name__
        model = self._model
        columns_id_str = self._columns_to_id.convert_columns_to_id(*self._columns)
        target_id_str = self._columns_to_id.convert_columns_to_id(self._target)
        saved_model_path = self._config.get_trained_models_path() + f'{model_type}_{columns_id_str}_{target_id_str}_model.sav'
        saved_model = pickle.dump(model, open(saved_model_path, 'wb'))

    def _load_model(self) -> Any:
        model_type = self.__class__.__name__
        columns_id_str = self._columns_to_id.convert_columns_to_id(*self._columns)
        target_id_str = self._columns_to_id.convert_columns_to_id(self._target)
        saved_model_path = self._config.get_trained_models_path() + f'{model_type}_{columns_id_str}_{target_id_str}_model.sav'
        if exists(saved_model_path):
            loaded_model = pickle.load(open(saved_model_path, 'rb'))
        else:
            raise Exception(f'modelo {model_type} não encontrado em {saved_model_path}. Treine o modelo antes de tentar carrega-lo.')

        return loaded_model

    @abstractmethod
    def evaluate(self, X_test:pd.DataFrame, Y_test:pd.DataFrame) -> np.ndarray:
        pass
