#!/usr/bin/env python
import os
import numpy
import spimage
import argparse


def center_image(filename, outfile, sigma=3):
    """Finds a localized strong object and puts it in the center. The
    sigma variable describes the size of the object"""
    print("foo")
    if not os.path.isfile(filename):
        raise IOError("Can not find file {0}".format(filename))
    img = spimage.sp_image_read(filename, 0)
    x = (numpy.arange(img.image.shape[0], dtype='float64')
         - img.image.shape[0]/2. + 0.5)
    y = (numpy.arange(img.image.shape[1], dtype='float64')
         - img.image.shape[1]/2. + 0.5)
    z = (numpy.arange(img.image.shape[2], dtype='float64')
         - img.image.shape[2]/2. + 0.5)
    kernel = numpy.exp(-(x[:, numpy.newaxis, numpy.newaxis]**2 +
                         y[numpy.newaxis, :, numpy.newaxis]**2 +
                         y[numpy.newaxis, numpy.newaxis, :]**2)/2.0/sigma**2)

    img_ft = numpy.fft.fft2(numpy.fft.fftshift(img.image))
    kernel_ft = numpy.fft.fft2(numpy.fft.fftshift(kernel))
    kernel_ft *= numpy.conj(img_ft)
    bt = numpy.fft.ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    min_z = 0
    for x in range(bt.shape[0]):
        for y in range(bt.shape[1]):
            for z in range(bt.shape[2]):
                if abs(bt[z, y, x]) > min_v:
                    min_v = abs(bt[z, y, x])
                    min_x = x
                    min_y = y
                    min_z = z
    print(min_x, min_y, min_z)
    spimage.sp_image_translate(img,
                               -(-min_z + bt.shape[0] // 2),
                               -(-min_y + bt.shape[1] // 2),
                               -(-min_x + bt.shape[2] // 2),
                               spimage.SP_TRANSLATE_WRAP_AROUND)
    shift = img

    spimage.sp_image_write(shift, outfile, 0)
    spimage.sp_image_free(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input file")
    parser.add_argument("outfile", help="Output file")
    parser.add_argument("-s", "--sigma", type=float, default=3,
                        help="Width of the particle that is centered in "
                        "pixels (optional, defaults to 3 pixels)")
    args = parser.parse_args()
    center_image(args.infile, args.outfile, args.sigma)
