#!/usr/bin/env python

from pylab import *
import os
import re
import sys

def copy_good(it_num, threshold):
    try:
        last_iteration = int(it_num)
    except:
        print "Error: \"%s\" is not a valid iteration number" % it_num
        sys.exit(1)
    try:
        threshold = float(threshold)
    except:
        print "Error \"%s\" is not a valid threshold" %threshold
        sys.exit(1)


    ls_out = os.popen('ls').readlines()

    expr = re.compile('[0-9]{6}')
    dirs = filter(expr.search,ls_out)
    dirs = [d[:-1] for d in dirs]

    for d in dirs:
        l = loadtxt("%s/uwrapc.log" % (d), skiprows=47)
        ferr = l[-1][3]

        if ferr < threshold:
            os.system("cp %s/real_space-%.7d.h5 final_good/%.3d.h5" % (d,last_iteration,int(d)))

if __name__ == "__main__":
    try:
        copy_good(int(sys.argv[1]),float(sys.argv[2]))
    except:
        print "Usage: copy_good <last iteration number> <threshold>"

