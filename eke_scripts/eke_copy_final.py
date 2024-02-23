#!/usr/bin/env python
import os
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Input directory")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()

    copy_final(args.indir, args.outdir)


def copy_final(in_dir, out_dir):
    all_dirs = os.listdir(in_dir)

    dirs = [this_dir for this_dir in all_dirs
            if re.search("^[0-9]{6}$", this_dir)]
    dirs.sort()

    all_files = os.listdir(f"{in_dir}/{dirs[0]}")

    files = [this_file for this_file in all_files
             if re.search("^real_space-[0-9]{7}.h5$", this_file)]
    files.sort()

    final_file = files[-1]
    print(f"Using file {final_file}")

    for d in dirs:
        try:
            os.system(f"cp {in_dir}/{d}/{final_file} {out_dir}/{int(d):04}.h5")
        except:
            print(f"Problem copying file from {d}")


if __name__ == "__main__":
    main()
