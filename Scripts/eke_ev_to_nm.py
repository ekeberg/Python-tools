#!/usr/bin/env python
import argparse
from eke import conversions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("energy", type=float)
    args = parser.parse_args()

    print("{0} nm".format(conversions.ev_to_nm(args.energy)))
