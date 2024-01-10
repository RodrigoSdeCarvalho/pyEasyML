import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)


# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

import pandas as pd
import numpy as np
import sklearn.model_selection, sklearn.preprocessing
from Utils.ColumnsToID import ColumnsToID 
from Configs.Config import Config
import pickle
from os.path import exists
from typing import Any
import glob
from pandas.core.indexes.base import Index as pdIndexes


class DataPreprocessor:
    def __init__(self) -> None:
        self._config = Config()
        self._columns_to_id = ColumnsToID()
        self._DATASET_PATH:str = ""
        self._DATA_FOLDER_PATH:str = self._config.get_data_path()

    @property
    def DATA_FOLDER_PATH(self) -> str:
        return self._DATA_FOLDER_PATH

    @property
    def DATASET_PATH(self) -> str:
        return self._DATASET_PATH

    @DATASET_PATH.setter
    def DATASET_PATH(self, dataset_file_name:str) -> None:
        self._DATASET_PATH = os.path.join(self._DATA_FOLDER_PATH, dataset_file_name)

    def read_dataset(self, path: str) -> pd.DataFrame:
        file_extension = os.path.splitext(path)[1].lower()
    
        if file_extension == ".csv":
            return pd.read_csv(path)
        elif file_extension == ".parquet":
            return pd.read_parquet(path)
        elif file_extension == ".xlsx":
            return pd.read_excel(path)
        # Add more conditions for other file formats if needed
        else:
            raise ValueError("Unsupported file extension: " + file_extension)    

    def save_dataset(self, dataset:pd.DataFrame, path:str) -> None:
        file_extension = os.path.splitext(path)[1].lower()
    
        if file_extension == ".csv":
            dataset.to_csv(path, index=False)
        elif file_extension == ".parquet":
            dataset.to_parquet(path, index=False)
        elif file_extension == ".xlsx":
            dataset.to_excel(path, index=False)
        # Add more conditions for other file formats if needed
        else:
            raise ValueError("Unsupported file extension: " + file_extension)   

    def read_all_data(self) -> pd.DataFrame:
        all_data_path = glob.glob(os.path.join(self._DATA_FOLDER_PATH, "all_data.*"))[0]

        if exists(all_data_path):
            print("Reading all_data.")
            return self.read_dataset(all_data_path)
        else:
            dataset_files = os.listdir(self._DATA_FOLDER_PATH)
            dataset_paths = [os.path.join(self._DATA_FOLDER_PATH, dataset_file) for dataset_file in dataset_files]
            
            datasets = []
            for dataset_path in dataset_paths:
                dataset = self.read_dataset(dataset_path)
                datasets.append(dataset)
            
            datasets = tuple(datasets)
            
            dataset = self.concat_datasets(*datasets)
            
            dataset.to_parquet(all_data_path)
        
        return dataset

    def concat_datasets(self, *datasets:pd.DataFrame) -> pd.DataFrame:

        return pd.concat(datasets, ignore_index=True)

    def get_train_val_test_datasets(self, shuffle: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw_dataset = self.read_dataset(self.DATASET_PATH)

        #Shuffle the dataframe
        if shuffle:
            raw_dataset = raw_dataset.sample(frac=1, random_state=self._config.RANDOM_STATE)

        len_data = len(raw_dataset)

        train_val_dataset = raw_dataset[:int(len_data*0.8)]
        test_dataset = raw_dataset[int(len_data*0.8):]

        return train_val_dataset, test_dataset

    def clean_dataset(self, *dfs:tuple[pd.DataFrame]) -> tuple[pd.DataFrame, ...]:
        # CLEAN THE DATASET HERE.

        return dfs

    def handle_missing_values(self, dataset:pd.DataFrame) -> pd.DataFrame:
        # HANDLE MISSING VALUES HERE.
        
        X = dataset.columns.to_list()
        for x in X:
            if dataset[x].isna().all():
                dataset.drop(labels=[x], axis=1, inplace=True)
            elif dataset[x].isna().any():
                dataset[x] = dataset[x].fillna(dataset[x].mean())

        return dataset

    def gen_train_test_datasets(self, dataset:pd.DataFrame, columns:pdIndexes = None, target_column = None, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if columns is None:
            columns = self._config.SELECTED_FEATURES
        else:
            columns = columns.to_list()

        if target_column is None:
            target_column = self._config.TARGET_FEATURE

        if target_column in columns:
            X = dataset[columns].drop(labels=[self._config.TARGET_FEATURE], axis=1)
            columns.remove(self._config.TARGET_FEATURE)
        else:
            X = dataset[columns]
        Y = dataset[target_column]

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            X.to_numpy(), 
            Y.to_numpy(), 
            test_size=0.3, 
            shuffle=shuffle,
            random_state=self._config.RANDOM_STATE
        )

        X_train, X_test = self.normalize_data([X_train, X_test], columns)

        return X_train, X_test, Y_train, Y_test

    def normalize_data(self, args:list[np.ndarray], columns:list[str]) -> tuple[np.ndarray, ...]:
        scaler = self.load_standard_scaler(columns)

        normalized_datasets = []
        for dataset in args:
            scaler.transform(dataset)
            normalized_datasets.append(scaler.transform(dataset))

        return tuple(normalized_datasets)

    def load_standard_scaler(self, columns:list[str]) -> sklearn.preprocessing.StandardScaler:
        columns_id_str = self._columns_to_id.convert_columns_to_id(*columns)

        scaler_path = os.path.join(self._config.get_normalization_model_path(),f'scaler_{columns_id_str}.sav')
        if exists(scaler_path):
            scaler = pickle.load(open(scaler_path, 'rb'))
        else:
            scaler = self.fit_standard_scaler(columns)

        return scaler

    def fit_standard_scaler(self, columns:list[str]) -> sklearn.preprocessing.StandardScaler:
        raw_dataset = self.read_dataset(self.DATASET_PATH)

        raw_dataset = raw_dataset[columns]
        raw_dataset = raw_dataset.to_numpy()

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(raw_dataset)

        columns_id_str = self._columns_to_id.convert_columns_to_id(*columns)
        path = os.path.join(self._config.get_normalization_model_path(), f'scaler_{columns_id_str}.sav')
        pickle.dump(scaler, open(path, 'wb'))

        return scaler

if __name__ == '__main__':
    #TESTE AS FUNÇÕES AQUI.
    pass
