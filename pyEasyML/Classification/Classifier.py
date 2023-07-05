import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

import pandas as pd
import numpy as np
from typing import Any
from os.path import exists
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, roc_curve, RocCurveDisplay, auc
from matplotlib import pyplot as plt

from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config

from Classification.Models.AbstractModel import AbstractModel
from Classification.Factory import Factory

class Classifier:
    """ Class used to implement the classification models.
        It can be used with a similar syntax as the sklearn models, 
        with the advantage of just needing to set the model name to load it.
        Check the available models in the BaseModels class.
        To use non-default params, just pass them as kwargs in the constructor.
        It also saves the trained models in the trained_models folder
        and is able to load them if they already exist.
        It's integrate to the Config and ColumnsToID class.
    """
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
        model = Factory().create(self._model_name, **params)

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

    #TODO : IMPLEMENT
    def cross_validate(self, dataset:pd.DataFrame, folds:int = 10) -> None:
        return self._model.cross_validate(dataset, folds)
    
    #TODO : IMPLEMENT
    def grid_search(self, dataset:pd.DataFrame, params:dict[str, Any]) -> None:
        return self._model.grid_search(dataset, params)

    #TODO : IMPLEMENT
    def random_search(self, dataset:pd.DataFrame, params:dict[str, Any]) -> None:
        return self._model.random_search(dataset, params)
    
    def plot_roc_curve(self, X_test:pd.DataFrame, Y_test:pd.DataFrame) -> None:
        class_labels = [0, 1]  # Assuming binary classification
        pr_data = []
        y_scores = self._model.predict_proba(X_test)[:, 1]  # Assuming probability scores for class 1
        precision, recall, _ = precision_recall_curve(Y_test, y_scores, pos_label=1)
        average_precision = sum(precision) / len(class_labels)
        average_recall = sum(recall) / len(class_labels)
        pr_data.append((self._model.__class__.__name__, precision, recall, average_precision, average_recall))

        # Plotting all Precision-Recall curves on a single figure
        plt.figure()
        for model_name, precision, recall, _, _ in pr_data:
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
            display.plot(ax=plt.gca())

        plt.legend([model_name for model_name, _, _, _, _ in pr_data])
        plt.title("2-class Precision-Recall curve")
        plt.show()
        