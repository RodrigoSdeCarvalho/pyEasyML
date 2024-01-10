import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)
sys.path.append(script_dir)

import pandas as pd
from Utils.Singleton import singleton
from Configs.Config import Config
from os.path import exists


@singleton
class ColumnsToID:
    def __init__(self) -> None:
        self._config = Config()
        self._CONVERT_DF_PATH = os.path.join(self._config.get_utils_path(), 'columnsToID.csv')

    def __load(self) -> pd.DataFrame:
        if exists(self._CONVERT_DF_PATH):
            return pd.read_csv(self._CONVERT_DF_PATH, index_col=0)
        else:
            return pd.DataFrame(columns=['column', 'id'])
    
    def convert_columns_to_id(self, *columns) -> int:
        df = self.__load()
        converted_columns = []
        for column in columns:
            if column not in df['column'].values:
                new_row = pd.DataFrame({'column': [column], 'id': [len(df)+1]})
                df = pd.concat([df, new_row], ignore_index=True)
                #df = df.append({'column': column, 'id': len(df)}, ignore_index=True)
                df.to_csv(self._CONVERT_DF_PATH)
                converted_columns.append(len(df) - 1)
            else:
                column_id = df[df['column'] == column]['id'].values[0]
                if column_id not in converted_columns:
                    converted_columns.append(column_id)

        converted_columns = sorted(converted_columns)

        return hash(int(''.join([str(column) for column in converted_columns])))

    def get_id(self, model_name:str) -> tuple[str]:
        models_path = self._config.get_trained_models_path()
        trained_models = os.listdir(models_path)

        for trained_model in trained_models:
            if model_name in trained_model:
                splitted_file_name = trained_model.split('_')
                columns_id, target_id, _ = tuple(splitted_file_name[1:])

                return columns_id, target_id
        
        raise Exception(f"Model {model_name} not found.")

    # TODO:TEST
    def column_id(self, column:str) -> int:
        df = self.__load()
        if column not in df['column'].values:
            raise Exception(f"Column {column} not found.")
        else:
            return df[df['column'] == column]['id'].values[0]

if __name__ == '__main__':
    cols = ColumnsToID()
    cols.convert_columns_to_id('d')
