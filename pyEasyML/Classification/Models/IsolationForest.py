import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

import numpy as np
from os.path import exists
from Utils import Definitions
from Data.DataPreprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest as IF
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel


class IsolationForest(AbstractClassificationModel):
    """Um modelo de classificação que utiliza o algoritmo IsolationForest. 
    Classifica os dados como outliers ou normais, sendo a classe normal a classe dominante.
    """
    def __init__(self, dominant_class:str = 'normal', unbalance_coeficient:float=0.4, columns:list[str]=DataPreprocessor.CLASSIFICATION_VAR_LIST) -> None:
        self.__dominant_class = dominant_class
        self.__unbalance_coeficient = unbalance_coeficient
        super().__init__(columns=columns)

    def _instantiate_model(self) -> IF:
        return IF(contamination=self.__unbalance_coeficient, random_state=Definitions.random_state)

    def train(self) -> bool:
        try:
            X_train, X_test, Y_train, Y_test = DataPreprocessor.gen_unbalanced_train_test_datasets(self._dominant_class, self._unbalance_coeficient, self._columns)
            print("Treinando modelo...")
            self._model.fit(X_train)
            self._save_model()
            print('Modelo treinado.')
            self._evaluate()

            return True
        except Exception as e:
            print(e)
            return False

    def _evaluate(self) -> None:
        X_train, X_test, Y_train, Y_test = DataPreprocessor.gen_unbalanced_train_test_datasets(self._dominant_class, self._unbalance_coeficient, self._columns)

        predictions = self.predict(X_test)

        print(accuracy_score(Y_test, predictions))        
        cm = confusion_matrix(Y_test, predictions)
        print(cm)
        print(classification_report(Y_test, predictions))

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        self._model:IF = self._load_model()
        class_probabilities = self.predict_proba(dataset)
        predictions = np.argmax(class_probabilities, axis=1)

        return predictions

    def predict_proba(self, dataset: np.ndarray) -> np.ndarray:
        self._model:IF = self._load_model()

        abnormality_scores = self._model.decision_function(dataset)
        min_score_reference, max_score_reference = self.get_min_max_score_reference(abnormality_scores)
        
        score_to_probability = np.vectorize(self.score_to_probability)
        healthy_scores = abnormality_scores[abnormality_scores > 0]
        anomaly_scores = abnormality_scores[abnormality_scores < 0]

        healthy_probabilities = score_to_probability(healthy_scores, min_score_reference, max_score_reference)
        anomaly_probabilities = score_to_probability(anomaly_scores, min_score_reference, max_score_reference)
        
        return np.stack((healthy_probabilities, anomaly_probabilities)).T

    def get_min_max_score_reference(self, scores:np.ndarray) -> float:
        scores.sort()

        min_score = np.mean(scores[0:5])
        max_score = np.mean(scores[-5:])

        return min_score, max_score

    def score_to_probability(self, score:float, min_score:float, max_score:float) -> float:
        if score < min_score:
            return 1

        if score > max_score:
            return 0

        return (max_score - score) / (max_score - min_score)
