#!/usr/bin/env python
import numpy
import os
import re
import sys
import argparse

def copy_good(last_iteration, threshold, in_dir, out_dir, prefix, error_type="fourier"):
    column_numbers = {"fourier" : 3, "real" : 2}
    if not error_type in column_numbers.keys():
        print("Error: invalid error_type")
        exit(1)
    ls_out = os.listdir(in_dir)
    dirs = [f for f in ls_out if re.search("[0-9]{6}", f)]

    if prefix:
        log = file("{0}/{1}.log".format(out_dir, prefix), "wp")
    else:
        log = file("{0}/copy_good.log".format(out_dir), "wp")

    count = 0
    for d in dirs:
        l = loadtxt("{0}/uwrapc.log".format(d), skiprows=47)

        err [i] = l[-1][column_numbers[error_type]]

        if err < threshold:
            os.system("cp {0}/real_space-{1:07}.h5 {2}/{3}{4:04}.h5".format(d,last_iteration,out_dir,
                                                                            prefix,count))
            log.write("{0} {1}/real_space-{2:07}.h5\n".format(count, d, last_iteration))
        count += 1
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=int, help="Iteration number to use")
    parser.add_argument("threshold", type=float, help="The error threshold")
    parser.add_argument("-i", "--indir", default=".", help="Input directory (default is .)")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory (default is .)")
    parser.add_argument("-p", "--prefix", default="", help="Optional prefix to outputed files")
    parser.add_argument("-e", "--error_type", default="fourier", help="Type of error to use: fourier or real")
    args = parser.parse_args()

    copy_good(args.file, args.threshold, args.indir, args.output_dir, args.prefix, args.error_type)


