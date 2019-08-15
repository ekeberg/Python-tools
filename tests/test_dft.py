from eke import dft
import numpy
import unittest

class TestDft(unittest.TestCase):
    def test_dft_1d(self):
        image_side = 10
        sample = numpy.random.random(image_side)
        ft_true = numpy.fft.fft(sample)
        dft_matrix = dft.dft_1d(image_side)
        ft_new = dft_matrix.dot(sample).squeeze()
        numpy.testing.assert_array_almost_equal(ft_true, ft_new, decimal=6)
        
    def test_dft_2d(self):
        shape = (10, 20)
        sample = numpy.random.random(shape)
        ft_true = numpy.fft.fft2(sample)
        dft_matrix = dft.dft_2d(shape[0], shape[1])
        ft_new = dft_matrix.dot(sample.flatten()).reshape(shape)
        numpy.testing.assert_array_almost_equal(ft_true, ft_new, decimal=6)

    def test_dft_nd_for_3d(self):
        shape = (5, 10, 15)
        sample = numpy.random.random(shape)
        ft_true = numpy.fft.fftn(sample)
        dft_matrix = dft.dft_nd(shape)
        ft_new = dft_matrix.dot(sample.flatten()).reshape(shape)
        numpy.testing.assert_array_almost_equal(ft_true, ft_new, decimal=6)

    def test_dft_nd_real_for_3d(self):
        shape = (5, 10, 15)
        sample = numpy.random.random(shape)
        ft_true = numpy.fft.fftn(sample)
        dft_matrix = dft.dft_nd_real(shape)
        ft_new = dft_matrix.dot(sample.flatten()).reshape(shape + (2, ))
        ft_new = (ft_new[..., 0::2] + 1.j*ft_new[..., 1::2]).reshape(shape)
        numpy.testing.assert_array_almost_equal(ft_true, ft_new, decimal=6)

    def test_dft_nd_masked_for_1d(self):
        shape = (10, )
        mask_real = numpy.bool8(numpy.random.randint(2, size=shape))
        mask_fourier = numpy.bool8(numpy.random.randint(2, size=shape))
        dft_full = dft.dft_nd(shape)
        dft_partial = dft.dft_nd_masked(mask_real, mask_fourier)
        numpy.testing.assert_array_almost_equal(
            dft_partial, dft_full[mask_fourier.flatten()][:, mask_real.flatten()])

    def test_dft_nd_masked_for_3d(self):
        shape = (5, 10, 15)
        mask_real = numpy.bool8(numpy.random.randint(2, size=shape))
        mask_fourier = numpy.bool8(numpy.random.randint(2, size=shape))
        dft_full = dft.dft_nd(shape)
        dft_partial = dft.dft_nd_masked(mask_real, mask_fourier)
        numpy.testing.assert_array_almost_equal(
            dft_partial, dft_full[mask_fourier.flatten()][:, mask_real.flatten()])

    def test_dft_nd_masked_real_for_3d(self):
        shape = (5, 10, 15)
        mask_real = numpy.bool8(numpy.random.randint(2, size=shape))
        mask_fourier = numpy.bool8(numpy.random.randint(2, size=shape))
        dft_full = dft.dft_nd_real(shape)
        dft_partial = dft.dft_nd_masked_real(mask_real, mask_fourier)
        dft_full_real = dft_full[0::2, :]
        dft_full_imag = dft_full[1::2, :]
        dft_partial_real = dft_partial[0::2, :]
        dft_partial_imag = dft_partial[1::2, :]
        numpy.testing.assert_array_almost_equal(
            dft_partial_real, dft_full_real[mask_fourier.flatten()][:, mask_real.flatten()])
        numpy.testing.assert_array_almost_equal(
            dft_partial_imag, dft_full_imag[mask_fourier.flatten()][:, mask_real.flatten()])

if __name__ == "__main__":
    unittest.main()
