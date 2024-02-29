from eke import rotmodule
import numpy
import unittest


class TestRotmodule(unittest.TestCase):
    def test_random(self):
        """Check that all quaternions have non-negative x-component and that
        the average is close to the x-axis"""
        nrots = 100
        rots = rotmodule.random(number_of_quaternions=nrots)
        self.assertGreaterEqual(rots[:, 0].min(), 0)
        numpy.testing.assert_array_almost_equal(rots.mean(axis=0)[1:],
                                                numpy.array((0., 0., 0.,)),
                                                decimal=1)

    def test_from_angle_and_dir(self):
        rot = rotmodule.from_angle_and_dir(0, (1, 0, 0))
        numpy.testing.assert_array_almost_equal(
            rot, numpy.array([1., 0., 0., 0.]))
        rot = rotmodule.from_angle_and_dir(numpy.pi, (0, 1, 0))
        numpy.testing.assert_array_almost_equal(
            rot, numpy.array([0., 0., 1., 0.]))

    def test_inverse(self):
        # Single
        rot_1 = rotmodule.from_angle_and_dir(numpy.pi/8., (1., 0., 0.))
        rot_2 = rotmodule.from_angle_and_dir(-numpy.pi/8., (1., 0., 0.))
        rot_1_inv = rotmodule.inverse(rot_1)
        numpy.testing.assert_array_almost_equal(rot_1_inv, rot_2)
        # Array
        rots = rotmodule.random(number_of_quaternions=10)
        inv_1 = rotmodule.inverse(rots)
        inv_2 = rots.copy()
        inv_2[:, 0] = -inv_2[:, 0]
        rotmodule.fix_sign(inv_2)
        numpy.testing.assert_array_almost_equal(inv_1, inv_2)

    def test_fix_sign(self):
        # Single
        rot = numpy.array([-1., 0., 0., 0.])
        rotmodule.fix_sign(rot)
        self.assertGreaterEqual(rot[0], 0)
        rot = numpy.array([0., 0., 0., -1.])
        rotmodule.fix_sign(rot)
        numpy.testing.assert_array_almost_equal(
            rot, numpy.array([0., 0., 0., 1.]))
        # Array
        rots = rotmodule.random(100, fix_sign=False)
        rotmodule.fix_sign(rots)
        self.assertGreaterEqual(rots[:, 0].min(), 0)

    def test_normalize(self):
        # Single
        rot = numpy.random.random(4)
        rot = rotmodule.normalize(rot)
        self.assertAlmostEqual(numpy.sqrt((rot**2).sum()), 1.)
        # Array
        rots = numpy.random.random((10, 4))
        rots = rotmodule.normalize(rots)
        numpy.testing.assert_array_almost_equal(
            numpy.sqrt((rots**2).sum(axis=1)), numpy.ones(10))

    def test_multiply(self):
        # Single
        rot = rotmodule.random()
        inv = rotmodule.inverse(rot)
        identity = numpy.array((1., 0., 0., 0.))
        numpy.testing.assert_array_almost_equal(
            rotmodule.multiply(rot, inv), identity)
        # Array
        rot = rotmodule.random(number_of_quaternions=10)
        inv = rotmodule.inverse(rot)
        identity = numpy.zeros((10, 4))
        identity[:, 0] = 1.
        numpy.testing.assert_array_almost_equal(
            rotmodule.multiply(rot, inv), identity)

    def test_relative(self):
        direction = numpy.random.random(3)
        angle = numpy.random.uniform(0., numpy.pi)
        rot_1 = rotmodule.from_angle_and_dir(angle, direction)
        rot_2 = rotmodule.from_angle_and_dir(-angle, direction)
        self.assertAlmostEqual(rotmodule.quaternion_to_angle(
            rotmodule.relative(rot_1, rot_2)),
                         2.*angle)

    def test_rotate_array(self):
        # Single rot, Single point
        rot = rotmodule.from_angle_and_dir(numpy.pi/2, (1., 0., 0.))
        point = numpy.random.random(3)
        target = numpy.array((point[0], -point[2], point[1]))
        rotated = rotmodule.rotate(rot, point)
        numpy.testing.assert_array_almost_equal(rotated, target)
        # Single rot, Multiple points
        rot = rotmodule.from_angle_and_dir(numpy.pi/2, (1., 0., 0.))
        points = numpy.zeros((10, 3))
        points[:, 1] = numpy.arange(10)
        rotated = rotmodule.rotate(rot, points)
        target = numpy.zeros((10, 3))
        target[:, 2] = numpy.arange(10)
        numpy.testing.assert_array_almost_equal(rotated, target)


if __name__ == "__main__":
    unittest.main()
