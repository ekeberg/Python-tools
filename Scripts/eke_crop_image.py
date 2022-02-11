#!/usr/bin/env python
"""Script to crop an h5 image and pad with zeros around it"""
from __future__ import print_function
import numpy
import spimage
from eke import image_manipulation


def crop_image(in_file, out_file, side, center=None):
    """Function to crop an h5 image and pad with zeros around it"""

    img = spimage.sp_image_read(in_file, 0)

    if not center:
        center = img.detector.image_center[:2]

    shifted = 0
    if img.shifted:
        shifted = 1
        img = spimage.sp_image_shift(img)

    cropped = spimage.sp_image_alloc(side, side, 1)
    cropped.image[:, :] = image_manipulation.crop_and_pad(
        img.image, center, side)
    cropped.mask[:, :] = image_manipulation.crop_and_pad(
        img.mask, center, side)

    if shifted:
        cropped = spimage.sp_image_shift(cropped)

    spimage.sp_image_write(cropped, out_file, 16)

    spimage.sp_image_free(img)
    spimage.sp_image_free(cropped)

    print("end")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input file")
    parser.add_argument("outfile", help="Output file")
    parser.add_argument("side", type=int, help="New side in pixels.")
    parser.add_argument("-c", "--center", default=None,
                        help="Image center to crop around. If not given the "
                        "center specified in the image is used.")
    args = parser.parse_args()

    if args.center:
        center = args.center.split('x')
        if len(center) != 2:
            raise ValueError("crop_image: Center must be of the form form "
                             "NxN.")
        center = numpy.array([float(center[0]), float(center[1])])
    else:
        center = None

    crop_image(args.infile, args.outfile, args.side, center)
