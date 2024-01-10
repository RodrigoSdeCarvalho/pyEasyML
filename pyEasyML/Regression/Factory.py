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

from Regression.Models.LinearSVR import LinearSVR
from Regression.Models.Ridge import Ridge
from Regression.Models.XGBRegressor import XGBRegressor
from Utils.Factory import Factory as BaseFactory
from Utils.AbstractModel import AbstractModel


class Factory(BaseFactory):
    _models:dict[str, AbstractModel] = {
        "LinearSVR": LinearSVR,
        "Ridge": Ridge,
        "XGBRegressor": XGBRegressor
    }
