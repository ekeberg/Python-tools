
from pylab import *
import os
import re
import sys

def get_errors():
    ls_out = os.listdir('.')
    dirs = [d for d in ls_out if re.search('^[0-9]{6}$', d)]

    ferr = zeros(len(dirs))

    for i in range(len(dirs)):
        l = loadtxt("%s/uwrapc.log" % (dirs[i]), skiprows=47)
        ferr[i] = l[-1][3]

    ferr.sort()
    for i in ferr:
        print i

if __name__ == "__main__":
    get_errors()
