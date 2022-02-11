import importlib.util as _util
import os as _os


def import_module(filename):
    """Load and return a specific python file as a module, even if
    the file is not in your pythonpath or current dir"""
    module_name = _os.path.splitext(_os.path.split(filename)[1])[0]
    spec = _util.spec_from_file_location(module_name, filename)
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_path(module_name):
    spec = _util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Can not import {module_name}")
    else:
        return spec.origin
