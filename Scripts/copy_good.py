#!/usr/bin/env python

from pylab import *
import os
import re
import sys

def copy_good(last_iteration, threshold, out_dir, prefix):
    ls_out = os.listdir('.')
    dirs = [f for f in ls_out if re.search('[0-9]{6}', f)]a

    log = open("%s/%s.log")
    count = 0
    for d in dirs:
        l = loadtxt("%s/uwrapc.log" % (d), skiprows=47)
        ferr = l[-1][3]

        if ferr < threshold:
            os.system("cp %s/real_space-%.7d.h5 %s/%.4d.h5" % (d,last_iteration,count),prefix)
            log.write("%d %s/real_space-%.7d.h5" % (count, d, last_iteration))
        count += 1

if __name__ == "__main__":
    parser = OptionParser(usage="%prog filename -f iteration_number -e threshold -o output_dir -p prefix")
    parser.add_option("-f", action="store", type="int", dest="file",
                      help="Iteration number to use")
    parser.add_option("-t", action="store", type="float", dest="threshold",
                      help="The error threshold")
    parser.add_option("-i", action="store", type="string", dest="input_dir", default=".",
                      help="Input directory (default is .)")
    parser.add_option("-o", action="store", type="string", dest="output_dir", default=".",
                      help="Output directory")
    parser.add_option("-p", action="store", type="string", dest="prefix", default="",
                      help="Optional prefix to outputed files")
    (options,args) = parser.parse_args()
    
    if len(args) < 1:
        parser.error("A filename must be specified")

    setup = 0
    try:
        setup = Setup(options.wavelength, options.distance,
                      options.pixel_size, options.number)
    except:
        print "Error in arguments"
        exit(1)
    plot_prtf(args[0],setup)
    pylab.show()

    copy_good(options.file, options.threshold, options.out_dir, options.prefix)


