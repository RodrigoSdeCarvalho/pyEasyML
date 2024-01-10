

import pandas as pd
import numpy as np
from typing import Any
from os.path import exists
from matplotlib import pyplot as plt

from Utils.AbstractModelDecorator import AbstractModelDecorator
from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config

from Regression.Models.AbstractRegressionModel import AbstractRegressionModel
from Regression.Factory import Factory


class Regressor(AbstractModelDecorator):
    """ Class used to implement the regression models.
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

    def run_cross_validation(self, X_train, Y_train, cv: int = 3, results_path: str = None) -> None:
        self._model.run_cross_validation(X_train, Y_train, cv, results_path)

    def plot_regression_scores(self, Y_test, Y_pred,
                               savefig: bool = False,
                               fig_path: str = "",
                               plot: bool = True):
        self._model.plot_regression_scores(Y_test, Y_pred, savefig, fig_path, plot)
