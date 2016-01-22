"""Calculate weakly constrained modes for a Fourier phase
retrieval problem using singular value decomposition"""
from . import dft as _dft
import numpy as _numpy 

def calculate_modes(real_mask, fourier_mask, reality_constraint=False,
                    number_of_modes=None):
    """Calculate weakly constrained modes for a Fourier phase retrieval
    problem using singular value decomposition"""
    if real_mask.shape != fourier_mask.shape:
        raise ValueError("Masks must have the same shape")

    real_mask_shifted = _numpy.fft.fftshift(real_mask)
    fourier_mask_shifted = _numpy.fft.fftshift(fourier_mask)

    if reality_constraint:
        sub_matrix = _dft.dft_2d_masked_real(real_mask.shape[0],
                                            real_mask.shape[1],
                                            real_mask_shifted,
                                            fourier_mask_shifted)
    else:
        sub_matrix = _dft.dft_2d_masked(real_mask.shape[0],
                                       real_mask.shape[1],
                                       real_mask_shifted,
                                       fourier_mask_shifted)

    real_modes_raw, singular_values, fourier_modes_raw1 = _numpy.linalg.linalg.svd(sub_matrix, full_matrices=False)
    fourier_modes_raw.dtype = "complex128"
    inverse_singular_values = _numpy.sqrt(1.-singular_values**2)

    if not number_of_modes:
        number_of_modes = len(singular_values)

    real_modes_shifted = _numpy.zeros((number_of_modes,
                                      real_mask_shifted.shape[0],
                                      real_mask_shifted.shape[1]),
                                     dtype="complex128")
    real_modes_shifted[:, real_mask_shifted] = real_modes_raw.T[:number_of_modes]
    real_modes = _numpy.fft.fftshift(real_modes_shifted, axes=(1, 2))
    fourier_modes_shifted = _numpy.zeros((number_of_modes,
                                         fourier_mask_shifted.shape[0],
                                         fourier_mask_shifted.shape[1]),
                                        dtype="complex128")
    fourier_modes_shifted[:, fourier_mask_shifted] = fourier_modes_raw[:number_of_modes]
    fourier_modes = _numpy.fft.fftshift(fourier_modes_shifted, axes=(1, 2))

    return inverse_singular_values[:number_of_modes], real_modes, fourier_modes
