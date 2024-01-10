import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

from Classification.Models.AbstractClassificationModel import AbstractClassificationModel
from sklearn.svm import SVC as svc


class SVC(AbstractClassificationModel):
    def __init__(self) -> None:
        super().__init__()

    def _instantiate_model(self, **params) -> svc:
        svc_model = svc(**params)

        return svc_model
