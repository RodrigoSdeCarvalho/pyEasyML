import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

os.environ['KERAS_BACKEND'] = 'theano'

from typing import Any
from os.path import exists

from Configs import Config
from Utils import ColumnsToID
from Utils.Threshold import Threshold
from NeuralNetworks.AbstractANN import AbstractANN

from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
import numpy as np

class ConvolutionalAutoencoder(AbstractANN):
    pass