#!/usr/bin/env python
import os
import numpy
import re
import argparse


def copy_good(last_iteration, threshold, in_dir, out_dir, prefix,
              error_type="fourier"):
    column_numbers = {"fourier": 3, "real": 2}
    if error_type not in column_numbers.keys():
        print("Error: invalid error_type")
        exit(1)
    ls_out = os.listdir(in_dir)
    dirs = [f for f in ls_out if re.search("[0-9]{6}", f)]

    if prefix:
        log = open("{0}/{1}.log".format(out_dir, prefix), "wp")
    else:
        log = open("{0}/copy_good.log".format(out_dir), "wp")

    count = 0
    for d in dirs:
        uwrapc_log = numpy.loadtxt("{0}/uwrapc.log".format(d), skiprows=47)

        err[i] = uwrapc_log[-1][column_numbers[error_type]]

        if err < threshold:
            os.system(f"cp {d}/real_space-{last_iteration:07}.h5 "
                      f"{out_dir}/{out_dirprefix}{count:04}.h5")
            log.write(f"{count} {d}/real_space-{last_iteration:07}.h5\n")
        count += 1
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=int,
                        help="Iteration number to use")
    parser.add_argument("threshold", type=float,
                        help="The error threshold")
    parser.add_argument("-i", "--indir", default=".",
                        help="Input directory (default is .)")
    parser.add_argument("-o", "--outdir", default=".",
                        help="Output directory (default is .)")
    parser.add_argument("-p", "--prefix", default="",
                        help="Optional prefix to outputed files")
    parser.add_argument("-e", "--error_type", default="fourier",
                        help="Type of error to use: fourier or real")
    args = parser.parse_args()

    copy_good(args.file, args.threshold, args.indir, args.output_dir,
              args.prefix, args.error_type)


