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
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import cross_validate
import json
from matplotlib import pyplot as plt

from Utils.AbstractModel import AbstractModel


class AbstractRegressionModel(AbstractModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict[str, float]:
        Y_pred = self._load_model().predict(X_test)

        return {
            "mae": metrics.mean_absolute_error(Y_test, Y_pred),
            "rmse": metrics.mean_squared_error(Y_test, Y_pred),
            "maxerror": metrics.max_error(Y_test, Y_pred),
            "r2": metrics.r2_score(Y_test, Y_pred)
        }

    def run_cross_validation(self, X_train, Y_train, cv: int = 3, results_path: str = None):
        scores = cross_validate(estimator=self._model,
                                X=X_train,
                                y=Y_train,
                                cv=cv,
                                scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error",
                                         "max_error", "r2", "neg_mean_absolute_percentage_error"])

        mae = -scores["test_neg_mean_absolute_error"]
        mse = -scores["test_neg_root_mean_squared_error"]
        max_error = scores["test_max_error"]
        r2 = scores["test_r2"]
        mape = -scores["test_neg_mean_absolute_percentage_error"]

        print("MAE:", mae)
        print("MSE:", mse)
        print("Max error:", max_error)
        print("R2:", r2)
        print("MAPE:", mape)

        results = {
            "mae": -scores["test_neg_mean_absolute_error"].mean(),
            "mse": -scores["test_neg_root_mean_squared_error"].mean(),
            "max_error": scores["test_max_error"].mean(),
            "r2": scores["test_r2"].mean(),
            "mape": -scores["test_neg_mean_absolute_percentage_error"].mean(),
        }

        if results_path is not None:
            json.dump(results, open(results_path, "w"))

    def plot_regression_scores(self, Y_test, Y_pred,
                               savefig: bool = False,
                               fig_path: str = "",
                               plot: bool = True):
        error = Y_test-Y_pred

        fig, axes = plt.subplots(3, 1)

        fig.tight_layout()

        axes[0].plot(Y_test, label="Real Consume")
        axes[0].plot(Y_pred, label="Predicted Consume")
        axes[0].legend()
        axes[0].set_title("Test vs. prediction comparison by timestamp")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Fuel Consumption (l/h)")

        axes[1].plot(error)
        axes[1].set_title("Error by timestamp")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Fuel Consumption (l/h)")

        axes[2].scatter(Y_test, Y_pred)
        axes[2].plot(Y_test, Y_test, color="red")
        axes[2].set_title("Test vs. Prediction by test values")
        axes[2].set_xlabel("Test Fuel Consumption ($(l/h)2$)")
        axes[2].set_ylabel("Predicted Fuel Consumption ($(l/h)2$)")

        fig.suptitle(f"Regression scores")

        fig.set_size_inches(18.5, 10.5)

        if plot:
            plt.show()

        if savefig:
            plt.savefig(fig_path)
