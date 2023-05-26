# https://stackoverflow.com/questions/7139111/python-extension-methods

class extension_method:

    def __init__(self, obj, method):
        method_name = method.__name__
        setattr(obj, method_name, method)
        self.obj = obj
        self.method_name = method_name

    def __enter__(self):
        return self.obj

    def __exit__(self, type, value, traceback):
        # remove this if you want to keep the extension method after context exit
        delattr(self.obj, self.method_name)

class C:
    pass

def get_class_name(self):
    return self.__class__.__name__

with extension_method(C, get_class_name):
    assert hasattr(C, 'get_class_name') # the method is added to C
    c = C()
    print(c.get_class_name()) # prints 'C'

assert not hasattr(C, 'get_class_name') # the method is gone from C
