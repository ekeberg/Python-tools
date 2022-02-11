#!/usr/bin/env python
import os
import PyPDF2
import argparse


def merge_pdf_files(input_filename_list, output_filename):
    input_file_handles = [PyPDF2.PdfFileReader(f) for f in input_filename_list]

    merger = PyPDF2.PdfFileMerger()

    for h in input_file_handles:
        merger.append(h)

    merger.write(output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first_file")
    parser.add_argument("additional_files", nargs="+")
    parser.add_argument("outfile")
    args = parser.parse_args()

    if args.outfile is None:
        output_filename = os.path.splitext(args[0])[0]+"_joined.pdf"
        print("No output file specified. Output will be written "
              f"to {output_filename}")
    else:
        output_filename = args.outfile

    merge_pdf_files([args.first_file]+args.additional_files, output_filename)
