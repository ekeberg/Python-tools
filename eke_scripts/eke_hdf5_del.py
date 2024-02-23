import argparse
import re
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)

    args = parser.parse_args()

    file_search_pattern = r"^(.+\.h5)(/.+)?$"

    match = re.search(file_search_pattern, args.file)
    input_file, input_key = match.groups()

    if input_key is None:
        raise ValueError("Must provide a location in the hdf5 file: "
                        "file.h5/location")

    with h5py.File(input_file, "r+") as file_handle:
        if input_key not in file_handle.keys():
            raise ValueError(f"Dataset or group {input_key} doew not exist "
                            f"in {input_file}")
        del file_handle[input_key]


if __name__ == "__main__":
    main()
