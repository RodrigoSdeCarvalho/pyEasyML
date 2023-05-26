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
    config = Config()

    @staticmethod
    def set_target_feature(target_feature:str) -> None:
        config = Settings.config
        config.TARGET_FEATURE = target_feature

    @staticmethod    
    def set_selected_features(selected_features:list[str]) -> None:
        config = Settings.config
        config.SELECTED_FEATURES = selected_features

    @staticmethod    
    def set_random_state(random_state:int) -> None:
        config = Settings.config
        config.RANDOM_STATE = random_state

    @staticmethod    
    def get_target_feature() -> str:
        config = Settings.config
        return config.TARGET_FEATURE

    @staticmethod
    def get_selected_features() -> list[str]:
        config = Settings.config
        return config.SELECTED_FEATURES

    @staticmethod
    def get_random_state() -> int:
        config = Settings.config
        return config.RANDOM_STATE
    