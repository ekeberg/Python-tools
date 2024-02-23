#!/usr/bin/env python
import argparse
from eke import conversions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("energy", type=float)
    args = parser.parse_args()

    wavelength = conversions.ev_to_nm(args.energy)

    print("{wavelength} nm")


if __name__ == "__main__":
    main()
