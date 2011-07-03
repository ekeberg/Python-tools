"""Module that contains all the scripts."""


import sys as _sys
import os as _os
import re as _re

_sys.path.append(_os.path.expanduser("~/Python/Scripts"))

_l = _os.listdir(_os.path.expanduser("~/Python/Scripts"))
if "scripts.py" in _l: _l.remove("scripts.py")
#print l
_scripts = [_f[:-3] for _f in _l if _re.search("\.py$",_f)]

for _f in _scripts:
    try:
        exec("import %s" % _f)
    except:
        print "Could not load %s" % _f
