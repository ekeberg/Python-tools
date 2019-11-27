#!/usr/bin/env python
import os
import sys
import h5py
import numpy
from eke import sphelper
import argparse

def split_pnccd(filename):
    with h5py.File(filename, "r") as file_handle:
        data2 = file_handle["data"]["data1"][...]

    base_name = os.path.splitext(os.path.split(filename)[-1])[0]
    sphelper.save_spimage("{0}_part1.h5".format(base_name), data2[:,:(data2.shape[1]//2)])
    sphelper.save_spimage("{0}_part2.h5".format(base_name), data2[:,(data2.shape[1]//2):])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    
    split_pnccd(args.file)
