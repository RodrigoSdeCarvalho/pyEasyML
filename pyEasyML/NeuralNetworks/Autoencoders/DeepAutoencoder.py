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

from NeuralNetworks.AbstractANN import AbstractANN
from NeuralNetworks.Layers.DeepAutoencoderLayers import DeepAutoencoderLayers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Input
import numpy as np

class DeepAutoencoder(AbstractANN):
    def __init__(self, 
                 inputs:int,
                 encode_layers_kwargs:dict[str, Any],
                 decode_layers_kwargs:dict[str, Any],
                 optimizer:str="adam",
                 loss:str="mse",
                 input_layer:Input=None,
                 layers:DeepAutoencoderLayers=None, 
                 columns_id=0) -> None:
        """
        Args:
            inputs (int): Number of features.
            encode_layers_kwargs (dict[str, Any]): all kwargs to be passed to the Dense layers of the encoder. Except units!
            decode_layers_kwargs (dict[str, Any]): all kwargs to be passed to the Dense layers of the decoder. Except units!
            batch_size (int, optional): Size of the batch. Defaults to 32.
            epochs (int, optional): Epochs that the model will be trained for. Defaults to 100.
            columns_id (int, optional): Id to save and load the model. Defaults to 0.
        """
        super().__init__(columns_id, target_id=0)
        self._model = self._instantiate_model(inputs=inputs, 
                                              encode_layers_kwargs=encode_layers_kwargs, 
                                              decode_layers_kwargs=decode_layers_kwargs,
                                              input_layer=input_layer,
                                              layers=layers,
                                              optimizer=optimizer,
                                              loss=loss)

    def _instantiate_model(self,
                           inputs:int, 
                           encode_layers_kwargs:dict[str, Any], 
                           decode_layers_kwargs:dict[str, Any],
                           input_layer:Input,
                           layers:DeepAutoencoderLayers,
                           optimizer:str,
                           loss:str) -> Sequential:
        # tries to load the model from disk
        model = self._load_model()
        if model is not None:
            return model

        # if the model is not found, it will be created. Gets the layers based on inputs and kwargs
        if layers is None and input_layer is None:
            input_layer, layers = self.__get_autoencoder_layers(inputs, encode_layers_kwargs, decode_layers_kwargs)

        # creates the model
        autoencoder = Sequential()
        
        # adds the layers to the model
        autoencoder.add(input_layer)
        for layer in layers:
            autoencoder.add(layer)

        # compiles the model
        autoencoder.compile(optimizer=optimizer, loss=loss)
        print("Input shape: ", autoencoder.input_shape)
        print(autoencoder.summary())

        return autoencoder

    def __get_autoencoder_layers(self, inputs:int, encode_layers_kwargs:dict[str, Any], decode_layers_kwargs:dict[str, Any]) -> tuple[Input, DeepAutoencoderLayers]:
        input_shape = (inputs,)
        num_features = np.prod(input_shape)
        latent_dim = num_features // 2

        # Input Layer
        input_layer = Input(shape=input_shape)

        # Encoding Layers
        layers = DeepAutoencoderLayers()

        # Adds the encoding layers
        units = num_features
        while units >= latent_dim:
            units //= 2 # Determines the number of units for the next layer
            layers.add(units=units, **encode_layers_kwargs)

        # Sets the position of the first dropout layer
        dropout_pos = len(layers) // 2

        # Adds the Decoding Layers
        units = layers.reverse()
        for unit in units:
            layers.add(units=unit, **decode_layers_kwargs)

        # Adds the dropout layers
        layers.insert_at(index=dropout_pos, name="Dropout", rate=0.2)
        layers.insert_at(index=len(layers)-dropout_pos+1, name="Dropout", rate=0.2)

        # Output Layer
        layers.add(units=num_features, **decode_layers_kwargs)

        return input_layer, layers

    def fit(self, X_train:np.ndarray, X_val:np.ndarray, callbacks:list[Any]=[EarlyStopping(monitor='val_loss')], epochs:int=100, batch_size:int=32, re_train=False) -> Any:
        history = self._model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        self._save_model()

        return history

    def predict(self, dataset:np.ndarray) -> Any:
        dataset = np.array(dataset)
        return self._model.predict(dataset)

    def evaluate(self, X_test:np.ndarray) -> Any:
        X_predicted = self.predict(X_test)
        mae = np.mean(np.abs(X_test - X_predicted), axis=1)
        
        return mae
