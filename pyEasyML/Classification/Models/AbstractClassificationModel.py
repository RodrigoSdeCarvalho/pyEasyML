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
from os.path import exists
from typing import Any
from sklearn import metrics
from sklearn.model_selection import cross_validate
import numpy as np

from Utils.AbstractModel import AbstractModel


class AbstractClassificationModel(AbstractModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, X_test:pd.DataFrame, Y_test:pd.DataFrame) -> np.ndarray:
        predictions = self._load_model().predict(X_test)
 
        cm = confusion_matrix(Y_test, predictions)
        report = classification_report(Y_test, predictions)
        print(report)

        return cm

    def run_cross_validation(self, X_train: np.ndarray, Y_train: np.ndarray, cv: int = 3) -> list[float]:
        scores = cross_validate(self._model, X_train, Y_train,
                                cv=3, scoring=["accuracy", "precision", "recall"])
        print("Accuracy:", scores["test_accuracy"])
        print("Precision:", scores["test_precision"])
        print("Recall:", scores["test_recall"])
