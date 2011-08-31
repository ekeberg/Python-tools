
import sys, os, re, pwd

home = os.path.expanduser('~')
#home = "/home/%s" % pwd.getpwuid(os.getuid())[0]
path = "%s/Python/Scripts" % home

#python_version = "/usr/bin/python2.6"
python_version = "/usr/local/bin/python64"

#l = os.popen("find %s" % path).readlines()
#files = [f[:-1] for f in l]

#expr = re.compile('.py$')
#py_files = filter(expr.search,files)
l = os.listdir(path)
if "scripts.py" in l: l.remove("scripts.py")
files = ["%s/%s" % (path,f) for f in l if re.search("\.py$",f)]
#print files

#expr = re.compile('^(.(?!home/ekeberg/Scripts/global))*$')
#expr = re.compile('^(.(?!%s/global))*$' % path)
#expr = re.compile('(?!^/home/ekeberg/Scripts/global)')
#py_files2 = filter(expr.search,py_files)

#py_files2.remove('%s/scripts.py' % path)

expr = re.compile('/\w+\.py')

names = []

for i in files:
    m = expr.search(i)
    names.append(i[m.start()+1:m.end()-3])

if not os.path.isdir("%s/global" % path):
    os.mkdir("%s/global" % path)


# for i in range(len(files)):
#     os.system("cp %s %s/global/python_script_%s" % 
#               (files[i],path,names[i]))

for i in range(len(files)):
    os.system("echo \"#! %s\" > %s/global/python_script_%s" % (python_version,path,names[i]))
    os.system("cat %s >> %s/global/python_script_%s" % (files[i],path,names[i]))

os.system("chmod 744 %s/global/*" % path)

expr = re.compile('%s/global' % path)
if not expr.search(os.environ['PATH']):
    print "It's suggested to add %s/global to your PATH." % path

