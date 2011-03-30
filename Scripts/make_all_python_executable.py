#! /usr/bin/python

import sys, os, re, pwd

home = os.path.expanduser('~')
#home = "/home/%s" % pwd.getpwuid(os.getuid())[0]

l = os.popen("find %s/Scripts" % home).readlines()
files = [f[:-1] for f in l]

expr = re.compile('.py$')
py_files = filter(expr.search,files)

#expr = re.compile('^(.(?!home/ekeberg/Scripts/global))*$')
expr = re.compile('^(.(?!%s/Scripts/global))*$' % home)
#expr = re.compile('(?!^/home/ekeberg/Scripts/global)')
py_files2 = filter(expr.search,py_files)

py_files2.remove('%s/Scripts/scripts.py' % home)

expr = re.compile('/\w+\.py')

names = []

for i in py_files2:
    m = expr.search(i)
    names.append(i[m.start()+1:m.end()-3])

if not os.path.isdir("%s/Scripts/global" % home):
    os.mkdir("%s/Scripts/global" % home)


for i in range(len(py_files2)):
    os.system("cp %s %s/Scripts/global/python_script_%s" % 
              (py_files2[i],home,names[i]))

os.system("chmod 744 %s/Scripts/global/*" % home)

expr = re.compile('%s/Scripts/global' % home)
if not expr.search(os.environ['PATH']):
    print "It's suggested to add %s/Scripts/global to your PATH." % home

