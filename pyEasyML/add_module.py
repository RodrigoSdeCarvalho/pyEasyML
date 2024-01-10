def add_pyEasyML_module():
    """
    Add pyEasyML module to python path.
        """
    import os
    import sys
    import re

    sys.dont_write_bytecode = True
    script_dir = os.path.abspath(__file__)
    script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)
    script_dir = os.path.abspath(script_dir)
    os.chdir(script_dir)
    sys.path.append(os.path.join(script_dir))
