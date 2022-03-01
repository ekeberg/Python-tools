from eke import tools
import numpy
import unittest


class TestTools(unittest.TestCase):
    def test_round_mask(self):
        """Compare output with predicted output. Use non-cube input"""
        reference = numpy.bool8([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                 [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
                                 [[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
                                 [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
                                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        result = tools.round_mask((5, 3, 4), 1.6)
        numpy.testing.assert_equal(reference, result)

    def test_downsample(self):
        before_sampling = numpy.random.random((20, 20, 20))
        after_sampling = tools.downsample(before_sampling, 3)
        self.assertEqual(after_sampling.shape, (6, 6, 6))
        self.assertAlmostEqual(before_sampling[:3, :3, :3].sum(),
                               after_sampling[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
