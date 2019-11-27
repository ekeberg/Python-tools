#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import argparse

sys.path.append('%s/Python/Modules' % os.path.expanduser('~'))

from eke import conversions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("energy", type=float)
    args = parser.parse_args()

    print("{0} nm".format(conversions.ev_to_nm(args.energy)))

