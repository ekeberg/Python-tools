"""Module that contains all the scripts of the Python-tools.
This is kind of deprecated and omly works on my laptop. Reusable parts
from scripts should be put in modules instead."""
import os as _os
import re as _re

_BLACKLIST = ['plot_image_3d']


def get_script_list():
    """Return a list of stirngs with the names of available scripts."""
    file_list = _os.listdir(_os.path.expanduser("~/Work/Python-tools/Scripts"))
    if "scripts.py" in file_list:
        file_list.remove("scripts.py")
    scripts = [_os.path.splitext(file_name)[0]
               for file_name in file_list
               if _re.search(r"\.py$", file_name)]
    return scripts


_SCRIPTS = get_script_list()
NOT_LOADED = []


for script in _SCRIPTS:
    if script not in _BLACKLIST:
        try:
            exec("import %s" % script)
        except:
            NOT_LOADED.append(script)
