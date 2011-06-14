
from pylab import *
import os
import re
import sys
from optparse import OptionParser

def copy_good(last_iteration, threshold, in_dir, out_dir, prefix, error_type="fourier"):
    column_numbers = {"fourier" : 3, "real" : 2}
        if not error_type in column_numbers.keys():
            print "Error: invalid error_type"
            exit(1)
    ls_out = os.listdir(in_dir)
    dirs = [f for f in ls_out if re.search('[0-9]{6}', f)]

    if prefix:
        log = file("%s/%s.log" % (out_dir, prefix), "wp")
    else:
        log = file("%s/copy_good.log" % out_dir, "wp")

    count = 0
    for d in dirs:
        l = loadtxt("%s/uwrapc.log" % (d), skiprows=47)

        err [i] = l[-1][column_numbers[error_type]]

        if err < threshold:
            os.system("cp %s/real_space-%.7d.h5 %s/%s%.4d.h5" % (d,last_iteration,out_dir,
                                                                 prefix,count))
            log.write("%d %s/real_space-%.7d.h5\n" % (count, d, last_iteration))
        count += 1
    log.close()


if __name__ == "__main__":
    parser = OptionParser(usage="%prog -f iteration_number -e threshold -o output_dir -p prefix")
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
    parser.add_option("-e", action="store", type="string", dest="error_type", default="fourier",
                      help="Type of error to use: fourier or real")
    (options,args) = parser.parse_args()

    copy_good(options.file, options.threshold, options.input_dir, options.output_dir, options.prefix, options.error_type)


