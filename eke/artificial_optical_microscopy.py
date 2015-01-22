import numpy


def dic(image, dic_angle, dic_phase, dic_split):
    """Differential interference contrast"""

    x = numpy.fft.fftshift((numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)/image.shape[0])[numpy.newaxis, :]
    y = numpy.fft.fftshift((numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)/image.shape[1])[:, numpy.newaxis]

    kx = dic_split*2.*numpy.pi*numpy.cos(dic_angle)
    ky = dic_split*2.*numpy.pi*numpy.sin(dic_angle)

    T = 1.+numpy.exp(1.j*(dic_phase - kx*x - ky*y))

    image_ft = numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity
    
def zer(image, zer_phase, zer_radius):
    """Zernicke phase contrast. Radius in range (0, 0.5)."""
    x = numpy.fft.fftshift((numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)/image.shape[0])[numpy.newaxis, :]
    y = numpy.fft.fftshift((numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)/image.shape[1])[:, numpy.newaxis]

    #T = 1.+numpy.exp(1.j*(dic_phase - kx*x - ky*y))
    T = numpy.ones(image.shape, dtype="complex128")
    T[:, :] = numpy.exp(1.j*zer_phase)
    T[x**2 + y**2 < zer_radius**2] = 1.

    image_ft = numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity

def sch(image, sch_angle):
    """Schlieren microscopy."""
    x = numpy.fft.fftshift((numpy.arange(image.shape[0])-image.shape[0]/2.+0.5)/image.shape[0])[numpy.newaxis, :]
    y = numpy.fft.fftshift((numpy.arange(image.shape[1])-image.shape[1]/2.+0.5)/image.shape[1])[:, numpy.newaxis]

    #T = 1.+numpy.exp(1.j*(dic_phase - kx*x - ky*y))
    # T = numpy.ones(image.shape, dtype="complex128")
    # T[:, :] = numpy.exp(1.j*zer_phase)
    # T[x**2 + y**2 < zer_radius**2] = 1.
    T = y < x*numpy.tan(sch_angle)

    image_ft = numpy.fft.fft2(image)
    image_ft *= T
    image_transformed  = numpy.fft.ifft2(image_ft)
    intensity = abs(image_transformed)**2
    return intensity
