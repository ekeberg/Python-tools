"""Library of symmetries. So far only a small number of point
group symmetries. Functions return the rotation quaternions that
relate symmetric elements."""

import numpy as _numpy
from . import rotations as _rotmodule
from . import icosahedral_sphere as _icosahedral_sphere


def rotational(fold, axis=(0., 0., 1.)):
    """Rotational symmetry around a specified axis."""
    rots = [_rotmodule.quaternion_from_dir_and_angle(2.*_numpy.pi *
                                                    float(this_fold) /
                                                    float(fold), axis)
            for this_fold in range(fold)]
    for this_rot in rots:
        _rotmodule.quaternion_fix_sign(this_rot)
    return _numpy.float64(rots)


def icosahedral():
    """Icosahedral (or 543) symmetry."""
    five_fold_axes = _numpy.array(_icosahedral_sphere.icosahedron_vertices())
    all_five_fold = _numpy.array([rotational(5, axis)
                                 for axis in five_fold_axes]).reshape((60, 4))

    three_fold_axes = _numpy.array(_icosahedral_sphere.icosahedron_faces()).mean(axis=1)
    all_three_fold = _numpy.array([rotational(3, axis)
                                  for axis in three_fold_axes]).reshape((60, 4))

    two_fold_axes = _numpy.array(_icosahedral_sphere.icosahedron_edges()).mean(axis=1)
    all_two_fold = _numpy.array([rotational(2, axis)
                                for axis in two_fold_axes]).reshape((60, 4))

    all_rotations = _numpy.concatenate((all_five_fold,
                                       all_three_fold,
                                       all_two_fold))

    def rotation_in_list(rotation, rotation_list, accuracy=0.01):
        """Check if rotation is in rotation_list. This comparison is aware
        of that we are dealing with rotations and accepts rotations within
        an angle specified by the accuracy."""
        for this_rotation in rotation_list:
            rel_angle = _rotmodule.quaternion_to_angle(_rotmodule.quaternion_relative(this_rotation,
                                                                                    rotation))
            rel_angle = min(abs(rel_angle), abs(rel_angle-2.*_numpy.pi))
            if rel_angle < accuracy:
                return True
        return False

    def prune_rotations(rotations):
        """Returns the same rotations but with duplicates removed."""
        for this_rotation in rotations:
            _rotmodule.quaternion_fix_sign(this_rotation)
        pruned_rotations = []
        for this_rotation in rotations:
            if not rotation_in_list(this_rotation, pruned_rotations):
                pruned_rotations.append(this_rotation)
        return _numpy.array(pruned_rotations)

    pruned_rotations = prune_rotations(all_rotations)

    # Ugly rounding to make numbers look better
    pruned_rotations = _numpy.float64(_numpy.float16(pruned_rotations))

    return pruned_rotations
