#!/usr/bin/env python
import sys
import h5py
from eke import sphelper
import argparse

def pnccd_to_image(infile, outfile):
    with h5py.File(infile, "r") as file_handle:
        i1 = list(file_handle.keys()).index('data')
        i2 = list(file_handle.values()[i1].keys()).index('data1')
        data = file_handle.values()[i1].values()[i2].value        

    sphelper.save_spimage(outfile, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Name of image to convert.")
    parser.add_argument("outfile", help="Writes output to this file.")
    args = parser.parse_args()
    
    pnccd_to_image(args.infile, args.outfile)


