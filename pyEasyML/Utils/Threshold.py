import os, sys, re
from typing import Any

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

import pandas as pd
from Configs.Config import Config
from os.path import exists

class Threshold():
    def __init__(self, model:str, column_id:int = 0, value:float=None) -> None:
        self.__config = Config()
        self.__THRESHOLD_DF_PATH = os.path.join(self.__config.get_utils_path(), 'thresholds.csv')
        self.__df = self.__load()
        self.__model = model
        self.__column_id = column_id

        if value is not None:
            self.__value = value
        else:
            self.__value = self.__get()
            
    def __load(self) -> pd.DataFrame:
        if exists(self.__THRESHOLD_DF_PATH):
            return pd.read_csv(self.__THRESHOLD_DF_PATH, index_col=0)
        else:
            df = pd.DataFrame(columns=['model', 'column_id', 'threshold'])
            df.to_csv(self.__THRESHOLD_DF_PATH, index=True)
            return df 

    def __save(self) -> None:
        self.__df.to_csv(self.__THRESHOLD_DF_PATH, index=True, index_label=False)

    def __get(self) -> float:
        if self.__df.empty:
            return 0
        else:
            threshold = self.__df[(self.__df['model'] == self.__model) & (self.__df['column_id'] == self.__column_id)]['threshold']
            if threshold.empty:
                return 0
            else:
                return threshold.values[0]

    def __set(self, value: float) -> None:
        self.__df = self.__load()
        
        df = self.__df[(self.__df['model'] == self.__model) & (self.__df['column_id'] == self.__column_id)]
        if df.empty:
            new_row = pd.DataFrame({'model': [self.__model], 'column_id': [self.__column_id], 'threshold': [value]})
            self.__df = pd.concat([self.__df, new_row], ignore_index=True)
        else:
            row_index = df.index[0]
            if self.__df.at[row_index, 'threshold'] != value:
                self.__df.at[row_index, 'threshold'] = value
            else:
                return

        self.__value = value
        self.__save()

    def __mul__(self, value: float) -> None:
        self.__value *= value
        return self

    def __str__(self) -> str:
        return f'Threshold: {self.__value}'

    def __repr__(self) -> str:
        return f'Threshold: {self.__value}'

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float) -> None:
        self.__set(value)

    @property
    def column_id(self) -> int:
        return self.__column_id

    @column_id.setter
    def column_id(self, column_id: int) -> None:
        self.__column_id = column_id
        self.__value = self.__get()

if __name__ == '__main__':
    pass
