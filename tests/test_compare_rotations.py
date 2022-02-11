from eke import rotmodule
from eke import compare_rotations
import numpy
import unittest


class TestCompareRotations(unittest.TestCase):
    def test_perturbation_no_symmetry_absolute(self):
        nrots = 100
        rotations_1 = rotmodule.random(number_of_quaternions=nrots)
        overall_rot = rotmodule.random()
        perturbation = 0.000001
        rotations_2 = rotmodule.normalize(
            rotations_1 + perturbation*numpy.random.random((nrots, 4)))
        repeated_overall_rot = numpy.repeat(overall_rot.reshape((1, 4)),
                                            nrots, axis=0)
        rotations_2 = rotmodule.multiply(repeated_overall_rot, rotations_2)
        avg_absolute = compare_rotations.absolute_orientation_error(
            rotations_1, rotations_2)
        self.assertLess(avg_absolute, 4.*numpy.pi/180.)

    def test_perturbation_no_symmetry_relative(self):
        nrots = 100
        rotations_1 = rotmodule.random(number_of_quaternions=nrots)
        overall_rot = rotmodule.random()
        perturbation = 0.000001
        rotations_2 = rotmodule.normalize(
            rotations_1 + perturbation*numpy.random.random((nrots, 4)))
        repeated_overall_rot = numpy.repeat(overall_rot.reshape((1, 4)),
                                            nrots, axis=0)
        rotations_2 = rotmodule.multiply(repeated_overall_rot, rotations_2)
        avg_relative = compare_rotations.relative_orientation_error(
            rotations_1, rotations_2)
        self.assertLess(avg_relative, 5.*numpy.pi/180.)

    def test_perturbation_with_symmetry_absolute(self):
        nrots = 100
        rotations_1 = rotmodule.random(number_of_quaternions=nrots)
        overall_rot = rotmodule.random()
        perturbation = 0.000001
        rotations_2 = rotmodule.normalize(
            rotations_1 + perturbation*numpy.random.random((nrots, 4)))
        symmetry_operations = ((1., 0., 0., 0.),
                               (0., 1., 0., 0.))
        symmetry_version = numpy.random.randint(2, size=nrots)
        symmetry_version[0] = 0
        for i in range(nrots):
            rotations_2[i, :] = rotmodule.multiply(
                symmetry_operations[symmetry_version[i]], rotations_2[i, :])
        repeated_overall_rot = numpy.repeat(overall_rot.reshape((1, 4)),
                                            nrots, axis=0)
        rotations_2 = rotmodule.multiply(repeated_overall_rot, rotations_2)
        avg_absolute = compare_rotations.absolute_orientation_error(
            rotations_1, rotations_2, symmetry_operations)
        self.assertLess(avg_absolute, 4.*numpy.pi/180.)

    def test_perturbation_with_symmetry_relative(self):
        nrots = 100
        rotations_1 = rotmodule.random(number_of_quaternions=nrots)
        overall_rot = rotmodule.random()
        perturbation = 0.000001
        rotations_2 = rotmodule.normalize(
            rotations_1 + perturbation*numpy.random.random((nrots, 4)))
        symmetry_operations = ((1., 0., 0., 0.),
                               (0., 1., 0., 0.))
        symmetry_version = numpy.random.randint(2, size=nrots)
        for i in range(nrots):
            rotations_2[i, :] = rotmodule.multiply(
                symmetry_operations[symmetry_version[i]], rotations_2[i, :])
        repeated_overall_rot = numpy.repeat(overall_rot.reshape((1, 4)),
                                            nrots, axis=0)
        rotations_2 = rotmodule.multiply(repeated_overall_rot, rotations_2)
        avg_relative = compare_rotations.relative_orientation_error(
            rotations_1, rotations_2, symmetry_operations)
        self.assertLess(avg_relative, 5.*numpy.pi/180.)


if __name__ == "__main__":
    unittest.main()
