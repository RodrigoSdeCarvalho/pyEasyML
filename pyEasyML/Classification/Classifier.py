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

from Utils.AbstractModelDecorator import AbstractModelDecorator
from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config

from Classification.Factory import Factory


class Classifier(AbstractModelDecorator):
    """ Class used to implement the classification models.
        It can be used with a similar syntax as the sklearn models, 
        with the advantage of just needing to set the model name to load it.
        Check the available models in the BaseModels class.
        To use non-default params, just pass them as kwargs in the constructor.
        It also saves the trained models in the trained_models folder
        and is able to load them if they already exist.
        It's integrate to the Config and ColumnsToID class.
    """
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
        super().__init__(model_name, columns, target, default_columns, default_target, **model_params)

    def run_cross_validation(self, X_train: np.ndarray, Y_train: np.ndarray, cv: int = 3) -> None:
        return self._model.run_cross_validation(X_train, Y_train, cv)

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
        