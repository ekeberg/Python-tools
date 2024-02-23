#!/usr/bin/env python
import argparse
from eke import elements
from eke import conversions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wavelength", type=float,
                        help="Photon wavelength in nm.")
    parser.add_argument("-e", "--energy", type=float,
                        help="Photon energy in eV (takes presedence over "
                        "wavelength).")
    parser.add_argument("-m", "--material",
                        help="Material. Can be an element or one of water, "
                        "protein, virus, cell.")
    parser.add_argument("-d", "--density", type=float,
                        help="If the material is an element, the density has "
                        "to be provided.")
    args = parser.parse_args()
    if not args.material:
        print("Error: A material has to be specified.")
        exit(1)
    elif not (args.wavelength or args.energy):
        print("Error: Energy or wavelength has to be specified.")
        exit(1)

    if args.energy:
        energy = args.energy
    else:
        energy = conversions.nm_to_ev(args.wavelength)

    if args.material in elements.ELEMENTS:
        if not args.density:
            print("Error: density has to be provided for this material.")
            exit(1)
        kwargs = {args.material: 1}
        attenuation_length = elements.get_attenuation_length(
            energy, elements.Material(args.density, **kwargs))
    elif args.material.lower() in elements.MATERIALS:
        attenuation_length = elements.get_attenuation_length(
            energy, elements.MATERIALS[args.material.lower()])
    else:
        print("Error: invalid material.")
        exit(1)

    print(f"{attenuation_length*1e6} um")


if __name__ == "__main__":
    main()
