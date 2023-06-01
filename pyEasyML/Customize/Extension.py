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
    def __init__(self, cls, method):
        method_name = method.__name__
        setattr(cls, method_name, method)
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

if __name__ == "__main__":
    class C:
        def __init__(self, a) -> None:
            self._a = a

        def test(self):
            return "TEST"

    def test(self):
        return "TEST" + str(self._a)

    c = Extend(C, test)(a=6)
    print(c.test()) # TEST6
