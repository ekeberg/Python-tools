#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import argparse
from eke import conversions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wavelength", type=float, help="Wavelength [nm]")
    args = parser.parse_args()

    print("{0} eV".format(conversions.nm_to_ev(args.wavelength)))
