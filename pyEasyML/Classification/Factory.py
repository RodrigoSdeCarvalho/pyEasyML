import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from typing import Any

# Created models will be imported here.
from Classification.Models.AbstractClassificationModel import AbstractClassificationModel as AbstractModel

from Classification.Models.SVC import SVC
from Classification.Models.XGBClassifier import XGBClassifier
from Classification.Models.KMeans import KMeans
from Classification.Models.LogisticRegression import LogisticRegression
from Classification.Models.GradientBoostingClassifier import GradientBoostingClassifier

from Utils.Factory import Factory as BaseFactory


class Factory(BaseFactory):
    _models:dict[str, AbstractModel] = {
        "SVC": SVC,
        "XGBClassifier": XGBClassifier,
        "KMeans": KMeans,
        "LogisticRegression": LogisticRegression,
        "GradientBoostingClassifier": GradientBoostingClassifier
    }