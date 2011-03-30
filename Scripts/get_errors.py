#!/usr/bin/env python

from pylab import *
import os
import re
import sys

def get_errors():
    ls_out = os.popen('ls').readlines()

    expr = re.compile('[0-9]{6}')
    dirs = filter(expr.search,ls_out)
    dirs = [d[:-1] for d in dirs]

    ferr = zeros(len(dirs))

    for i in range(len(dirs)):
        l = loadtxt("%s/uwrapc.log" % (dirs[i]), skiprows=47)
        ferr[i] = l[-1][3]

    ferr.sort()
    for i in ferr:
        print i

if __name__ == "__main__":
    get_errors()
