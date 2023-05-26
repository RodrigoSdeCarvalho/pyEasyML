import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

from Classification.Models.AbstractModel import AbstractModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Configs import Config

class RandomForestClassifier(Model):
    def __init__(self, columns:list[str] ) -> None:
        super().__init__(columns)

    def _instantiate_model(self) -> RFC:
        rf_model = RFC(n_estimators=100, max_depth=2, random_state=self._config.RANDOM_STATE)

        return rf_model
