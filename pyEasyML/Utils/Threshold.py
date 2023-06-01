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
        self._config = Config()
        self._THRESHOLD_DF_PATH = os.path.join(self._config.get_utils_path(), 'thresholds.csv')
        self._df = self.__load()
        self._model = model
        self._column_id = column_id

        if value is not None:
            self._value = value
        else:
            self._value = self.__get()
            
    def __load(self) -> pd.DataFrame:
        if exists(self._THRESHOLD_DF_PATH):
            return pd.read_csv(self._THRESHOLD_DF_PATH, index_col=0)
        else:
            df = pd.DataFrame(columns=['model', 'column_id', 'threshold'])
            df.to_csv(self._THRESHOLD_DF_PATH, index=True)
            return df 

    def __save(self) -> None:
        self._df.to_csv(self._THRESHOLD_DF_PATH, index=True, index_label=False)

    def __get(self) -> float:
        if self._df.empty:
            return 0
        else:
            threshold = self._df[(self._df['model'] == self._model) & (self._df['column_id'] == self._column_id)]['threshold']
            if threshold.empty:
                return 0
            else:
                return threshold.values[0]

    def __set(self, value: float) -> None:
        self._df = self.__load()
        
        df = self._df[(self._df['model'] == self._model) & (self._df['column_id'] == self._column_id)]
        if df.empty:
            new_row = pd.DataFrame({'model': [self._model], 'column_id': [self._column_id], 'threshold': [value]})
            self._df = pd.concat([self._df, new_row], ignore_index=True)
        else:
            row_index = df.index[0]
            if self._df.at[row_index, 'threshold'] != value:
                self._df.at[row_index, 'threshold'] = value
            else:
                return

        self._value = value
        self.__save()

    def __mul__(self, value: float) -> None:
        self._value *= value
        return self

    def __str__(self) -> str:
        return f'Threshold: {self._value}'

    def __repr__(self) -> str:
        return f'Threshold: {self._value}'

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self.__set(value)

    @property
    def column_id(self) -> int:
        return self._column_id

    @column_id.setter
    def column_id(self, column_id: int) -> None:
        self._column_id = column_id
        self._value = self.__get()

if __name__ == '__main__':
    pass
