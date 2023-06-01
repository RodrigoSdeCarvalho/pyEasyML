import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

import json

def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper

@singleton
class Config():
    def __init__(self, client:str) -> None:
        self._configs:dict[str, str] = self.__load_configs(client)
        self._client:str = self._configs['client']
        self._SELECTED_FEATURES:list[str] = []
        self._TARGET_FEATURE:str = "NOT DEFINED"
        self._RANDOM_STATE:int = 0

    def __load_configs(self, client: str) -> dict[str, str]:
        start = client

        for tries in range(100):
            json_path = os.path.join(start, 'configs.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    configs = json.load(file)
                    return configs
            else:
                parent_dir = os.path.dirname(os.path.dirname(start))
                start = parent_dir

        return {'client': 'NOT DEFINED'}

    @property
    def SELECTED_FEATURES(self) -> list[str]:
        return self._SELECTED_FEATURES

    @SELECTED_FEATURES.setter
    def SELECTED_FEATURES(self, selected_features:list[str]) -> None:
        self._SELECTED_FEATURES = selected_features

    @property
    def TARGET_FEATURE(self) -> str:
        return self._TARGET_FEATURE
    
    @TARGET_FEATURE.setter
    def TARGET_FEATURE(self, target_feature:str) -> None:
        self._TARGET_FEATURE = target_feature

    @property
    def RANDOM_STATE(self) -> int:
        return self._RANDOM_STATE

    def get_data_path(self) -> str:
        return os.path.join(self._client, 'dataset/')

    def get_trained_models_path(self) -> str:
        return os.path.join(self._client, 'models/')

    def get_normalization_model_path(self) -> str:
        return os.path.join(self._client, 'models/', 'normalizationModel/')

    def get_utils_path(self) -> str:
        return os.path.join(self._client, 'Utils/')
    

if __name__ == '__main__':
    definitions = Config()
    print(definitions.TARGET_FEATURE)
    definitions.TARGET_FEATURE = 'target'

    definitions_2 = Config()
    definitions_2.TARGET_FEATURE = 'target_2'
    print(definitions.TARGET_FEATURE)

