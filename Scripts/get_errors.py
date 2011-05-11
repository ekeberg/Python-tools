
from pylab import *
import os
import re
import sys
from optparse import OptionParser

def get_errors(input_dir="."):
    ls_out = os.listdir(input_dir)
    dirs = [d for d in ls_out if re.search('^[0-9]{6}$', d)]

    ferr = zeros(len(dirs))

    for i in range(len(dirs)):
        l = loadtxt("%s/uwrapc.log" % (dirs[i]), skiprows=47)
        ferr[i] = l[-1][3]

    ferr.sort()
    for i in ferr:
        print i

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [-p DIRECTORY]")
    parser.add_option("-i", action="store", type="string", dest="input_dir", default=".",
                      help="The directory in which to search for reconstructions. Default is .")
    (options,argv) = parser.parse_args()
    if not os.path.isdir(options.input_dir):
        print "Error: %s is not a directory" % options.input_dir
        exit(1)
    get_errors(options.input_dir)
