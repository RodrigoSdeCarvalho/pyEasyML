import os, sys, re

# Evitando a criação de arquivos .pyc
sys.dont_write_bytecode = True

from typing import Any
import types

class Extend:
    def __init__(self, cls, method):
        method_name = method.__name__

        self._previous_method = getattr(cls, method_name, False)
        
        self._cls = cls
        self._cls_copy = types.new_class(cls.__name__, bases=(cls,))

        self._method_name = method_name
        self._method = method

        setattr(self._cls_copy, method_name, method)

    def __call__(self, *args, **kwargs) -> Any:
        """Triggered when the object is called.

        Returns:
            Any: Any object of any class that was inputed to the Extension.
        """
        return self._cls_copy(*args, **kwargs)

    def __enter__(self) -> Any:
        """Triggered when the object is entered. As in with statement.

        Returns:
            Any: Any object of any class that was inputed to the Extension.
        """
        setattr(self._cls, self._method_name, self._method)
        return self._cls

    def __exit__(self, type, value, traceback) -> None:
        """called when exiting the with statement.
            When exiting the with statement, a overwritten method is returned to the original state.
            And an added method is removed.
        """
        if self._previous_method:
            setattr(self._cls, self._method_name, self._previous_method)
        else:
            delattr(self._cls, self._method_name)

    @staticmethod
    def object(obj, method):
        cls = obj.__class__
        cls_copy = types.new_class(cls.__name__, bases=(cls,))
        setattr(cls_copy, method.__name__, method)

        return cls_copy(*obj.__dict__.values())

if __name__ == "__main__":
    class C:
        def __init__(self, a) -> None:
            self._a = a

        def test(self):
            return "TEST"

    def test(self):
        return "TEST" + str(self._a)

    c = C(6)
    with Extend(C, test):
        print(c.test()) # TEST6
    print(c.test()) # TEST

    c = Extend(C, test)(6)
    print(c.test()) # TEST6

    print(C(6).test()) # TEST
    
    a = C(15)
    a = Extend.object(a, test)
    print(a.test()) # TEST6
    print(C(6).test()) # TEST