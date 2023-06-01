import os, sys, re
from typing import Any

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

script_dir = os.path.abspath(__file__)

# Apagando o nome do arquivo e deixando apenas o diretorio.
script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)

os.chdir(script_dir)

sys.path.append(script_dir)

class Extend:
    def __init__(self, obj, method):
        method_name = method.__name__
        setattr(obj, method_name, method)
        self.obj = obj
        self.method_name = method_name

    def __enter__(self):
        return self.obj

    def __exit__(self, type, value, traceback):
        delattr(self.obj, self.method_name)

