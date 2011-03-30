#!/usr/bin/env python

import os
import re
import sys

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
    if len(sys.argv) != 3:
        print "Usage: copy_final.py <in_dir> <out_dir>"
        sys.exit(1)
    copy_final(sys.argv[1],sys.argv[2])
