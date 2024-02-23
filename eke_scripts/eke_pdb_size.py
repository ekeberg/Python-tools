import argparse
import warnings
import numpy
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("infile", help="Input file")
    args = arg_parser.parse_args()

    fname = args.infile

    pdb_parser = PDBParser()

    # Ignore PDB warnings, we are just interested in the size

    structure = pdb_parser.get_structure(0, fname)
    atoms = list(structure.get_atoms())

    natoms = len(atoms)

    coords = numpy.zeros((natoms, 3))

    for index, this_atom in enumerate(atoms):
        coords[index, :] = this_atom.get_vector().get_array()

    coords /= 10  # Convert from Ångström to nm.

    x_size, y_size, z_size = sizing_function(coords)

    # print("Warning: the result will depend on the orientation of the protein")
    print(f"x: {x_size} nm")
    print(f"y: {y_size} nm")
    print(f"z: {z_size} nm")


def get_cartesian_size(coords):
    x_size = coords[:, 0].max() - coords[:, 0].min()
    y_size = coords[:, 1].max() - coords[:, 1].min()
    z_size = coords[:, 2].max() - coords[:, 2].min()
    return x_size, y_size, z_size


def get_pca_size(coords):
    pca = sklearn.decomposition.PCA()
    coords_rotated = pca.fit_transform(coords)
    x_size = coords_rotated[:, 0].max() - coords_rotated[:, 0].min()
    y_size = coords_rotated[:, 1].max() - coords_rotated[:, 1].min()
    z_size = coords_rotated[:, 2].max() - coords_rotated[:, 2].min()
    return x_size, y_size, z_size


try:
    import sklearn.decomposition
    sizing_function = get_pca_size
    print("Printing size along principal components")
except ImportError:
    sizing_function = get_cartesian_size
    print("Printing size along xyz axes")


if __name__ == "__main__":
    main()
