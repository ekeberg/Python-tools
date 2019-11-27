import numpy as _numpy


def dic(image, dic_angle, dic_phase, dic_split):
    """Differential interference contrast"""

    x = (_numpy.fft.fftshift((_numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)
                             / float(image.shape[0]))[_numpy.newaxis, :])
    y = (_numpy.fft.fftshift((_numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)
                             / float(image.shape[1]))[:, _numpy.newaxis])

    kx = dic_split*2.*_numpy.pi*_numpy.cos(dic_angle)
    ky = dic_split*2.*_numpy.pi*_numpy.sin(dic_angle)

    T = 1.+_numpy.exp(1.j*(dic_phase - kx*x - ky*y))

    image_ft = _numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = _numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity
    
def zer(image, zer_phase, zer_radius):
    """Zernicke phase contrast. Radius in range (0, 0.5)."""
    x = (_numpy.fft.fftshift((_numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)
                             / float(image.shape[0]))[_numpy.newaxis, :])
    y = (_numpy.fft.fftshift((_numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)
                             / float(image.shape[1]))[:, _numpy.newaxis])

    #T = 1.+_numpy.exp(1.j*(dic_phase - kx*x - ky*y))
    T = _numpy.ones(image.shape, dtype="complex128")
    T[:, :] = _numpy.exp(1.j*zer_phase)
    T[x**2 + y**2 < zer_radius**2] = 1.

    image_ft = _numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = _numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity

def sch(image, sch_angle):
    """Schlieren microscopy."""
    x = (_numpy.fft.fftshift((_numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)
                             / float(image.shape[0]))[_numpy.newaxis, :])
    y = (_numpy.fft.fftshift((_numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)
                             / float(image.shape[1]))[:, _numpy.newaxis])

    T = y < x*_numpy.tan(sch_angle)

    image_ft = _numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = _numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity
