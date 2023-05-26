import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

import numpy as np
from Configs import Config
from Classification.Models.AbstractModel import AbstractModel
from sklearn.svm import OneClassSVM as ocSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class OneClassSVM(Model):
    """Um modelo de classificação que utiliza o algoritmo OneClassSVM. Classifica os dados como outliers ou normais, sendo a classe normal a classe dominante.
    """
    def __init__(self, columns:list[str] = None, dominant_class:str = 'normal', unbalance_coeficient:float=0.4) -> None:
        """Um modelo de classificação que utiliza o algoritmo OneClassSVM.
        Ele é utilizado para classificar os dados como outliers ou normais,
        sendo a classe normal a classe dominante.
        Args:
            dominant_class (str): Classe dominante.
            unbalance_coeficient (float): Coeficiente de desbalanceamento. Porcentagem da classe não-dominante em relação à classe dominante. É a proporção de outliers que entra como parametro no modelo.
        """
        pass
        self._dominant_class = dominant_class
        self._unbalance_coeficient = unbalance_coeficient
        super().__init__(columns=columns)

    def _instantiate_model(self) -> ocSVM:
        return ocSVM(nu=self._unbalance_coeficient, kernel="rbf", gamma='auto')

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
        self._model:ocSVM = self._load_model()
        class_probabilities = self.predict_proba(dataset)
        predictions = np.argmax(class_probabilities, axis=1)

        return predictions

    def predict_proba(self, dataset: np.ndarray) -> np.ndarray:
        self._model:ocSVM = self._load_model()
        hyperplane_distances = self._model.decision_function(dataset)

        dominant_class_hyperplane_distances = hyperplane_distances[hyperplane_distances > 0]
        non_dominant_class_hyperplane_distances = hyperplane_distances[hyperplane_distances < 0]

        dominant_max_distance_reference = self._get_max_distance_reference(dominant_class_hyperplane_distances)
        non_dominant_max_distance_reference = self._get_max_distance_reference(non_dominant_class_hyperplane_distances)

        distances = np.stack((hyperplane_distances, hyperplane_distances))
        probabilities = self._gen_probabilies(distances, dominant_max_distance_reference, non_dominant_max_distance_reference)

        return probabilities

    def _get_max_distance_reference(self, distances:np.ndarray) -> np.ndarray:
        distances = np.abs(distances)
        distances.sort()
        sorted_distances = distances[::-1] 
        largest_distances = sorted_distances[:5]
        max_distance_reference = np.mean(largest_distances)

        return max_distance_reference

    def _gen_probabilies(self, distances:np.ndarray, 
                                dominant_max_distance_reference:float, 
                                non_dominant_max_distance_reference:float) -> np.ndarray:
        normal_probabilities = distances[0]
        anomalous_probabilities = -distances[1]

        vectorized_distance_to_probability = np.vectorize(self._distance_to_probability)

        normal_probabilities = vectorized_distance_to_probability(normal_probabilities, dominant_max_distance_reference)
        anomalous_probabilities = vectorized_distance_to_probability(anomalous_probabilities, non_dominant_max_distance_reference)

        return np.stack((normal_probabilities, anomalous_probabilities)).T

    def _distance_to_probability(self, distance:float, max_distance_reference:float) -> float:
        if distance <= 0:
            return 0

        if distance >= max_distance_reference:
            return 1

        return distance / max_distance_reference
