import os, re

l = os.popen('ls ~/Scripts').readlines()

expr = re.compile('.py$')
files = filter(expr.search,l)

files = [f[:-1] for f in files]

print files

files.remove('scripts.py')
files.remove('make_all_python_executable.py')

print files

for f in files:
    try:
        m = __import__(f[:-3],globals(),locals(),[],-1)
        vars()[f[:-3]] = m
    except:
        "could not load module %s" % f[:-3]

del files,f,l,expr
del os,re


