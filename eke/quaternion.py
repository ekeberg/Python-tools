"""Tools to manipulate quaternions representing rotations. This is
not a tool for general quaternions."""
import numpy as _numpy


class Quaternion(object):
    """Rotation quaternion. Using automatic normalization"""
    def __init__(self, elements):
        if len(elements) != 4:
            raise ValueError("Quaternion must be initialzied by a lenght "
                             "4 array.")
        self._elements = _numpy.float64(elements)

        self._elements[:] /= _numpy.sqrt((self._elements**2).sum())

        if ((self._elements[0] < 0) or
                (self._elements[0] == 0 and self._elements[1] < 0) or
                (self._elements[0] == 0 and self._elements[1] == 0 and
                 self._elements[2] < 0) or
                (self._elements[0] == 0 and self._elements[1] == 0 and
                 self._elements[2] == 0 and self._elements[3] < 0)):
            self._elements[:] = -self._elements

    def __str__(self):
        return (f"Quaternion({self._elements[0]}, {self._elements[1]}, "
                f"{self._elements[2]}, {self._elements[3]})")
    __repr__ = __str__

    def __getitem__(self, index):
        if not isinstance(index, slice) and (index < 0 or index >= 4):
            raise IndexError(f"Index must be 0, 1, 2 or 3. Not {index}")
        return self._elements[index]

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply a Quaternion with another "
                            f"Quaternion. Not a {other}")
        return Quaternion((self[0]*other[0] - self[1]*other[1]
                           - self[2]*other[2] - self[3]*other[3],
                           self[0]*other[1] + self[1]*other[0]
                           + self[2]*other[3] - self[3]*other[2],
                           self[0]*other[2] - self[1]*other[3]
                           + self[2]*other[0] + self[3]*other[1],
                           self[0]*other[3] + self[1]*other[2]
                           - self[2]*other[1] + self[3]*other[0]))

    def __rmul__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply a Quaternion with another "
                            f"Quaternion. Not a {other}")
        return Quaternion((other[0]*self[0] - other[1]*self[1]
                           - other[2]*self[2] - other[3]*self[3],
                           other[0]*self[1] + other[1]*self[0]
                           + other[2]*self[3] - other[3]*self[2],
                           other[0]*self[2] - other[1]*self[3]
                           + other[2]*self[0] + other[3]*self[1],
                           other[0]*self[3] + other[1]*self[2]
                           - other[2]*self[1] + other[3]*self[0]))

    def __invert__(self):
        return Quaternion((self[0], -self[1], -self[2], -self[3]))


def angle(quaternion):
    """Get the rotation angle for a quaternion"""
    return 2.*_numpy.arccos(quaternion[0])


def axis(quaternion):
    """Get the axis of rotation for a quaternion"""
    return quaternion[1:] / _numpy.linalg.norm(quaternion[1:])


def random():
    """Get a random quaternion"""
    random_base = _numpy.random.random(3)
    return Quaternion((_numpy.sqrt(1.-random_base[0])
                       * _numpy.sin(2.*_numpy.pi*random_base[1]),
                       _numpy.sqrt(1.-random_base[0])
                       * _numpy.cos(2.*_numpy.pi*random_base[1]),
                       _numpy.sqrt(random_base[0])
                       * _numpy.sin(2.*_numpy.pi*random_base[2]),
                       _numpy.sqrt(random_base[0])
                       * _numpy.cos(2.*_numpy.pi*random_base[2])))


def from_angle_and_dir(angle, direction):
    """Create a Quaternion from an angle and a direction"""
    norm = _numpy.linalg.norm(direction)*_numpy.sin(angle/2.)
    return Quaternion((_numpy.cos(angle/2.),
                       direction[0]/norm,
                       direction[1]/norm,
                       direction[2]/norm))


def rotation_matrix(quaternion):
    """Get the rotation matrix corresponding to a quaternion"""
    return _numpy.matrix([[quaternion[0]**2 + quaternion[1]**2
                           - quaternion[2]**2 - quaternion[3]**2,
                           2.0*quaternion[1]*quaternion[2]
                           - 2.0*quaternion[0]*quaternion[3],
                           2.0*quaternion[1]*quaternion[3]
                           + 2.0*quaternion[0]*quaternion[2]],
                          [2.0*quaternion[1]*quaternion[2]
                           + 2.0*quaternion[0]*quaternion[3],
                           quaternion[0]**2 - quaternion[1]**2
                           + quaternion[2]**2 - quaternion[3]**2,
                           2.0*quaternion[2]*quaternion[3]
                           - 2.0*quaternion[0]*quaternion[1]],
                          [2.0*quaternion[1]*quaternion[3]
                           - 2.0*quaternion[0]*quaternion[2],
                           2.0*quaternion[2]*quaternion[3]
                           + 2.0*quaternion[0]*quaternion[1],
                           quaternion[0]**2 - quaternion[1]**2
                           - quaternion[2]**2 + quaternion[3]**2]])


def average(quaternion_list):
    """Generate the quaternion that best represent the average
    of a list of quaternions"""
    return Quaternion(_numpy.array(
        [this_quaternion[:]
         for this_quaternion in quaternion_list]).mean(axis=0))
