#!/usr/bin/env python
import argparse
from eke import conversions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavelength", type=float, help="Wavelength [nm]")
    args = parser.parse_args()

    energy = conversions.nm_to_ev(args.wavelength)

    print(f"{energy} eV")


if __name__ == "__main__":
    main()
