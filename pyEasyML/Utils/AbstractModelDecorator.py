

import pandas as pd
import numpy as np
from typing import Any
from os.path import exists
from matplotlib import pyplot as plt

from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config

from Utils.AbstractModel import AbstractModel
from Regression.Factory import Factory


class AbstractModelDecorator:
    ModelFactory = Factory

    def __init__(self,
                 model_name:str,
                 columns:list[str] = None,
                 target:str = None,
                 default_columns:bool = False,
                 default_target:bool = False,
                 **model_params:dict[str, Any]) -> None:
        """Sets the columns and the target to be used and instantiates the model.

        Args:
            model_name (str): name of the model to be used. Must be the same as the one in the BaseModels class.

            columns (list[str], optional): Colums to use as X. Defaults to None.

            target (str, optional): Column to use as Y. Defaults to None.

            default_columns (bool, optional): Flag that indicates the use of Config.SELECTED_FEATURES. Defaults to False.

            default_target (bool, optional): Flag that indicates the use of Config.TARGET. Defaults to False.
        """

        # Name of the model to be used
        self._model_name = model_name

        # Singleton Config object
        self._config = Config()

        # Singleton ColumnsToID object
        self._columns_to_id = ColumnsToID()

        # Columns to be used
        self._columns = self._get_columns(columns, default_columns)

        # Target to be used
        self._target = self._get_target(target, default_target)

        # Model to be used, chose from the base models
        self._model = self._get_model(**model_params)

    def _get_columns(self, columns:list[str], default_columns:bool) -> list[str]:
        if columns is None and default_columns:
            return self._config.SELECTED_FEATURES
        elif columns is None and not default_columns:
            raise Exception("Columns not set.")
        else:
            return columns

    def _get_target(self, target:str, default_target:bool) -> str:
        if target is None and default_target:
            return self._config.TARGET_FEATURE
        elif target is None and not default_target:
            raise Exception("Target not set.")
        else:
            return target

    def _get_model(self, **params:dict[str, Any]) -> AbstractModel:
        try:
            model = self._instantiate_model(**params)
            model.columns = self._columns
            model.target = self._target

            return model
        except:
            raise Exception(f"Model {self._model_name} not found. Implement it first.")

    def _instantiate_model(self, **params:dict[str, Any]) -> AbstractModel:
        model = self.ModelFactory.create(self._model_name, **params)

        return model

    @property
    def model(self) -> Any:
        return self._model

    @property
    def columns(self) -> list[str]:
        return self._columns

    @columns.setter
    def columns(self, columns:list[str]) -> None:
        self._columns = columns

    def fit(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, re_train:bool = False) -> bool:
        """Retorna True se o modelo foi treinado com sucesso ou se já havia sido treinado anteriormente.
           Retorna False se houve algum erro durante o treinamento.

        Returns:
            bool: _description_
        """
        try:
            if self._verify_if_model_trained():
                if re_train:
                    self._model.fit(X_train, Y_train)
                else:
                    print(f"Model {self._model_name} already trained for the columns given and the target {self._target}.")
                    return True
            else:
                return self._model.fit(X_train, Y_train)
        except Exception as e:
            print(e)
            return False

    def _verify_if_model_trained(self) -> bool:
        columns_id_str = self._columns_to_id.convert_columns_to_id(*self._columns)
        target_id_str = self._columns_to_id.convert_columns_to_id(self._target)

        if exists(self._config.get_trained_models_path() + f'{self._model_name}_{columns_id_str}_{target_id_str}_model.sav'):
            return True
        else:
            False

    def evaluate(self, X_test:pd.DataFrame, Y_test:pd.DataFrame) -> None:
        return self._model.evaluate(X_test, Y_test)

    def predict(self, dataset:pd.DataFrame) -> np.ndarray:
        return self._model.predict(dataset)

    def predict_proba(self, dataset:pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(dataset)
