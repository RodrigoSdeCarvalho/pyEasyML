import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)


os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from Configs.Config import Config
from Classification.Factory import Factory as CFactory
from Regression.Factory import Factory as RFactory
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from  sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from typing import Any, Callable
from pandas.core.indexes.base import Index as pdIndexes


class FeatureSelector:
    def __init__(self, *train_dfs:tuple[pd.DataFrame, pd.DataFrame], columns:pdIndexes, model_name:str=None) -> None:
        self._X_train, self._Y_train = train_dfs

        if model_name == None:
            self._model = None
        else:
            self._model = self._create_model(model_name=model_name)
        self._config = Config()
        
        if self._config.TARGET_FEATURE in columns:
            self._columns = columns.drop(self._config.TARGET_FEATURE)
        else:
            self._columns = columns

    def _create_model(self, model_name:str) -> Any:
        if CFactory.is_model_registered(model_name=model_name):
            return CFactory.create(model_name=model_name)
        elif RFactory.is_model_registered(model_name=model_name):
            return RFactory.create(model_name=model_name)
        else:
            print(f"Modelo {model_name} não encontrado.")
            raise KeyError(f"Modelo {model_name} não encontrado.")

    def select_percentile(self, func:Callable, percentile:int) -> list[str]:
        selector = SelectPercentile(score_func=func, percentile=percentile)
        selector.fit(self._X_train, self._Y_train)
        selected_columns = self._columns[selector.get_support()].values.tolist()

        return selected_columns

    def select_k_best(self, func:Callable, k:int) -> list[str]:
        selector = SelectKBest(score_func=func, k=k)
        selector.fit(self._X_train, self._Y_train)
        selected_columns = self._columns[selector.get_support()].values.tolist()

        return selected_columns

    def sequential_feature_selector(self, k:int, score_func_name='neg_mean_squared_error', direction='forward', cv=5) -> list[str]:
        selector = sfs(estimator=self._model.model, n_features_to_select=k, scoring=score_func_name, direction=direction, cv = cv)
        selector = selector.fit(self._X_train, self._Y_train)
        selected_columns = list(selector.get_feature_names_out())
        print(selected_columns)

        return selected_columns

    def recursive_feature_elimination(self, k:int, step=1, verbose=0) -> list[str]:
        selector = RFE(estimator=self._model.model, n_features_to_select=k, step=step, verbose=verbose)
        selector.fit(self._X_train, self._Y_train)
        selected_columns = self._columns[selector.get_support()].values.tolist()
        print(selected_columns)

        return selected_columns

    def recursive_feature_elimination_CV(self, k:int, cv=5) -> list[str]:
        selector = RFECV(estimator=self._model.model, min_features_to_select=k, cv=cv)
        selector.fit(self._X_train, self._Y_train)
        selected_columns = self._columns[selector.get_support()].values.tolist()
        print(selected_columns)
        
        return selected_columns

    def select_from_model(self, max_features:int, threshold:float) -> list[str]:
        selector = SelectFromModel(estimator=self._model.model, max_features=max_features, threshold=threshold)
        selector.fit(self._X_train, self._Y_train)
        selected_columns = self._columns[selector.get_support()].values.tolist()
        
        return selected_columns
