import pathlib
import argparse
from eke import shell_functions

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The files to remove should be at this "
                    "path + NUMBER.xxx")
args = parser.parse_args()

raw_path = pathlib.Path(args.path)

shell_functions.remove_all_but_last(raw_path)
