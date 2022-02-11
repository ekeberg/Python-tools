#!/usr/bin/env python
import spimage
import argparse


def image_to_png(in_filename, out_filename, colorscale, shift):
    try:
        img = spimage.sp_image_read(in_filename, 0)
    except IOError:
        raise TypeError(f"{in_filename} is not a readable file")

    if shift == 1:
        img = spimage.sp_image_shift(img)

    try:
        spimage.sp_image_write(img, out_filename, colorscale)
    except IOError:
        raise TypeError(f"Can not write {out_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 file to convert to png.")
    parser.add_argument("outfile", help="Output file.")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Use log scale.")
    parser.add_argument("-s", "--shift", action="store_true",
                        help="Shift image.")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Output mask.")
    parser.add_argument("-c", "--colorscale",
                        choices=("gray", "jet", "phase", "hot",
                                 "rainbow", "traditional", "weighted_phase"),
                        default="jet",
                        help="Colorscale of output image.")
    args = parser.parse_args()

    colorscale_dict = {"gray": spimage.SpColormapGrayScale,
                       "jet": spimage.SpColormapJet,
                       "phase": spimage.SpColormapPhase,
                       "hot": spimage.SpColormapHot,
                       "rainbow": spimage.SpColormapRainbow,
                       "traditional": spimage.SpColormapTraditional,
                       "weighted_phase": spimage.SpColormapWeightedPhase}

    output_flag = colorscale_dict[args.colorscale]
    if args.log:
        output_flag |= spimage.SpColormapLogScale
    if args.mask:
        output_flag |= spimage.SpColormapMask

    image_to_png(args.infile, args.outfile, output_flag, args.shift)
