#!/bin/env python
import os
import re
import sys
from optparse import OptionParser

def copy_final(in_dir,out_dir):
    l = os.listdir(in_dir)

    dirs = [d for d in l if re.search("^[0-9]{6}$",d)]
    dirs.sort()

    l = os.listdir("%s/%s" % (in_dir,dirs[0]))

    files = [i for i in l if re.search("^real_space-[0-9]{7}.h5$",i)]
    files.sort()

    final_file = files[-1]
    print "Using file %s" % final_file

    for d in dirs:
        try:
            os.system("cp %s/%s/%s %s/%.4d.h5" % (in_dir,d,final_file,out_dir,int(d)))
        except:
            print "Problem copying file from %s" % d


if __name__ == "__main__":
    parser = OptionParser(usage="%prog filename [options]")
    parser.add_option("-i", action="store", type="string", dest="indir",
                          help="Input directory")
    parser.add_option("-o", action="store", type="string", dest="outdir",
                          help="Output directory")
    (options,args) = parser.parse_args()

    if not (options.indir and options.outdir):
        parser.print_usage()
        sys.exit(1)
    copy_final(options.indir,options.outdir)
