import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

script_dir = os.path.abspath(script_dir)

os.chdir(script_dir)

sys.path.append(os.path.join(script_dir))

from Configs.Config import Config

class Settings:
    def __init__(self, client:str) -> None:
        self._config = Config(client)

    def set_target_feature(self,target_feature:str) -> None:
        config = self._config
        config.TARGET_FEATURE = target_feature

    def set_selected_features(self,selected_features:list[str]) -> None:
        config = self._config
        config.SELECTED_FEATURES = selected_features

    def set_random_state(self,random_state:int) -> None:
        config = self._config
        config.RANDOM_STATE = random_state

    def get_target_feature(self) -> str:
        config = self._config
        return config.TARGET_FEATURE

    def get_selected_features(self) -> list[str]:
        config = self._config
        return config.SELECTED_FEATURES

    def get_random_state(self) -> int:
        config = self._config
        return config.RANDOM_STATE
    