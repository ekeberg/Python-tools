def get_h5_in_dir(path):
    "Returns a list of all the h5 files in a directory"
    import os
    import re
    l = os.listdir(path)
    files = ["%s/%s" % (path,f) for f in l if re.search("\.h5$",f)]
    return files
    
def gaussian_blur(image, sigma):
    """Returns a blured version of the image. The kernel is a gaussian with radius sigma."""
    import pylab
    size = pylab.shape(image)
    x = pylab.arange(size[0]) - size[0]/2.0 + 0.5
    y = pylab.arange(size[1]) - size[1]/2.0 + 0.5
    z = pylab.arange(size[2]) - size[2]/2.0 + 0.5
    kernel = pylab.fftshift(pylab.exp(-2.0*sigma**2*pylab.pi**2*((x/size[0])**2 +
                                                                 (y[:,pylab.newaxis]/size[1])**2 +
                                                                 (z[:,pylab.newaxis,pylab.newaxis]/size[2])**2)))
    image_ft = pylab.fftn(image)
    image_ft*= kernel
    product = pylab.ifftn(image_ft)
    return product

def remove_duplicates(input_list):
    seen = []
    for index, value in enumerate(input_list):
        if value in seen:
            input_list.pop(index)
        else:
            seen.append(value)

def factorial(n):
    from scipy.special import gamma as gamma_
    return gamma_(n+1)

def bincoef(n, k):
    from scipy.special import binom
    return binom(n,k)
