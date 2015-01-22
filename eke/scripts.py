"""Module that contains all the scripts of the Python-tools."""
import os as _os
import re as _re

_BLACKLIST = ['plot_image_3d']

def get_script_list():
    """Return a list of stirngs with the names of available scripts."""
    file_list = _os.listdir(_os.path.expanduser("~/Work/Python/Scripts"))
    if "scripts.py" in file_list:
        file_list.remove("scripts.py")
    scripts = [_os.path.splitext(file_name)[0] for file_name in file_list if _re.search("\.py$", file_name)]
    return scripts

_SCRIPTS = get_script_list()
NOT_LOADED = []

for script in _SCRIPTS:
    if not script in _BLACKLIST:
        try:
            exec("import %s" % script)
        except:
            NOT_LOADED.append(script)
