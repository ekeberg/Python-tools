# Author: Tomas Ekeberg
# Dedicated to: Nicusor Timneanu

import argparse

ELEMENTS = ['Ru', 'Re', 'Ra', 'Rb', 'Rn', 'Rh', 'Be', 'Ba', 'Bi', 'Br', 'H', 'P', 'Os', 'Ge', 'Gd', 'Ga', 'Pr', 'Pt', 'C', 'Pb', 'Pa', 'Pd', 'Xe', 'Po', 'Pm', 'Ho', 'Hf', 'Hg', 'He', 'Mg', 'K', 'Mn', 'O', 'S', 'W', 'Zn', 'Eu', 'Zr', 'Er', 'Ni', 'Na', 'Nb', 'Nd', 'Ne', 'Fr', 'Fe', 'B', 'F', 'Sr', 'N', 'Kr', 'Si', 'Sn', 'Sm', 'V', 'Sc', 'Sb', 'Se', 'Co', 'Cl', 'Ca', 'Ce', 'Cd', 'Tm', 'Cs', 'Cr', 'Cu', 'La', 'Li', 'Tl', 'Lu', 'Th', 'Ti', 'Te', 'Tb', 'Tc', 'Ta', 'Yb', 'Dy', 'I', 'U', 'Y', 'Ac', 'Ag', 'Ir', 'Al', 'As', 'Ar', 'Au', 'At', 'In', 'Mo']

parser = argparse.ArgumentParser()
parser.add_argument("input_file",  type=str)
parser.add_argument("output_file",  type=str)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

new_lines = []
with open(input_file, "r") as file_handle:
    for line in file_handle.readlines():
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            element = line[12:14]
            if element.strip().title() not in ELEMENTS:
                element = line[13:15]
            if element.strip().title() not in ELEMENTS:
                raise ValueError(f"Unrecognized atom name: {line[12:16]}")
            line = line[:-1]
            line = line.ljust(76)[:76] + element + "  "
        else:
            line = line[:-1]
        new_lines.append(line)

with open(output_file, "w") as file_handle:
    file_handle.write("\n".join(new_lines))

