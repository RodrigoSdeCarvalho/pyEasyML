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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
