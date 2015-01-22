#!/bin/env python
import os
import PyPDF2
from optparse import OptionParser
from optparse import OptionGroup

def merge_pdf_files(input_filename_list, output_filename):
    input_file_handles = [PyPDF2.PdfFileReader(f) for f in input_filename_list]
    
    merger = PyPDF2.PdfFileMerger()

    for h in input_file_handles:
        merger.append(h)

    merger.write(output_filename)

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [-o OUT_FILE] FILE1 FILE2 FILE3 ...")
    parser.add_option("-o", action="store", type="string", dest="output_filename")
    (options,args) = parser.parse_args()
    
    if len(args) < 2:
        parser.error("At least two filenames must be specified")

    if options.output_filename == None:
        output_filename = os.path.splitext(args[0])[0]+"_joined.pdf"
        print "No output file specified. Output will be written to {0}".format(output_filename)
    else:
        output_filename = options.output_filename

    merge_pdf_files(args, output_filename)
