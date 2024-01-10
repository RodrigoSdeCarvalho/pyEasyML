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
from matplotlib import pyplot as plt

from Utils.AbstractModelDecorator import AbstractModelDecorator
from Utils.ColumnsToID import ColumnsToID
from Configs.Config import Config

from Regression.Models.AbstracRegressiontModel import AbstracRegressiontModel
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

    #TODO : IMPLEMENT
    def cross_validate(self, dataset:pd.DataFrame, folds:int = 10) -> None:
        return self._model.cross_validate(dataset, folds)
