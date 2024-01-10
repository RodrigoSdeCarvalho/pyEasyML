

from typing import Any

from NeuralNetworks.AbstractANN import AbstractANN
from NeuralNetworks.Layers.DeepAutoencoderLayers import DeepAutoencoderLayers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Input
import numpy as np
from keras import regularizers

class DeepAutoencoder(AbstractANN):
    def __init__(self, 
                 inputs:int,
                 layers_params:dict[str, Any],
                 encode_layers_kwargs:dict[str, Any],
                 decode_layers_kwargs:dict[str, Any],
                 latent_space_dim:int,
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
                                              layers_params=layers_params,
                                              encode_layers_kwargs=encode_layers_kwargs, 
                                              decode_layers_kwargs=decode_layers_kwargs,
                                              latent_space_dim=latent_space_dim,
                                              input_layer=input_layer,
                                              layers=layers,
                                              optimizer=optimizer,
                                              loss=loss)

    def _instantiate_model(self,
                           inputs:int,
                           layers_params:dict[str, Any],
                           encode_layers_kwargs:dict[str, Any], 
                           decode_layers_kwargs:dict[str, Any],
                           latent_space_dim:int,
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
            input_layer, layers = self.__get_autoencoder_layers(inputs, layers_params, encode_layers_kwargs, decode_layers_kwargs, latent_space_dim)

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

    def __get_autoencoder_layers(self, inputs:int, layers_params:dict[str, Any], encode_layers_kwargs:dict[str, Any], decode_layers_kwargs:dict[str, Any], latent_space_dim:int=None) -> tuple[Input, DeepAutoencoderLayers]:
        input_shape = (inputs,)
        num_features = np.prod(input_shape)
        
        if latent_space_dim is None:
            latent_space_dim = num_features // 2

        # Input Layer
        input_layer = Input(shape=input_shape)

        # Encoding Layers
        layers = DeepAutoencoderLayers()

        # Adds the encoding layers        
        units = num_features
        layer_ctr = 0
        while units >= latent_space_dim:
            layer_key = str(layer_ctr)
            if layer_key == '0' and layer_key in list(layers_params.keys()):
                layers.add(units=units, **layers_params[layer_key])
                layer_ctr += 1
                continue
            
            units //= 2 # Determines the number of units for the next layer
            if layer_key in list(layers_params.keys()):
                layers.add(units=units, **layers_params[layer_key])
            else:
                layers.add(units=units, **encode_layers_kwargs)
                
            layer_ctr += 1

        # Sets the position of the first dropout layer
        dropout_pos = len(layers) // 2

        # Adds the Decoding Layers
        units = layers.reverse()
        for unit in units:
            layer_key = str(layer_ctr)
            if layer_key in list(layers_params.keys()):
                layers.add(units=unit, **layers_params[layer_key])
            else:
                layers.add(units=unit, **decode_layers_kwargs)
            layer_ctr += 1

        # Adds the dropout layers
        layers.insert_at(index=dropout_pos, name="Dropout", rate=0.05)
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
