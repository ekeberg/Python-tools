import numpy as _numpy
import itertools as _itertools
from . import rotmodule as _rotmodule
from . import refactor


def relative_orientation_error(correct_rotations,
                               recovered_rotations,
                               symmetry_operations=((1., 0., 0., 0.), ),
                               shallow_ewald=False,
                               number_of_samples=1000):
    symmetry_operations = _numpy.array(symmetry_operations)
    # number_of_samples = 200 #3000
    number_of_rotations = len(recovered_rotations)
    average_angle = 0.
    for _ in range(number_of_samples):
        index_1 = _numpy.random.randint(number_of_rotations)
        index_2 = _numpy.random.randint(number_of_rotations)
        if index_1 == index_2:
            index_2 = (index_1 + 1) % number_of_rotations

        if shallow_ewald:
            friedel_rotation = [
                _rotmodule.from_angle_and_dir(0, (0, 0, 1)),
                _rotmodule.from_angle_and_dir(_numpy.pi, (0, 0, 1))
            ]
        else:
            friedel_rotation = [
                _rotmodule.from_angle_and_dir(0, (0, 0, 1))
            ]

        comparison_results = []
        for f1 in friedel_rotation:
            for f2 in friedel_rotation:
                for s in symmetry_operations:
                    this_rot = _rotmodule.multiply(
                        _rotmodule.inverse(correct_rotations[index_2]),
                        _rotmodule.inverse(s),
                        correct_rotations[index_1],
                        _rotmodule.inverse(f1),
                        _rotmodule.inverse(recovered_rotations[index_1]),
                        recovered_rotations[index_2],
                        f2
                    )

                    comparison_results.append(_rotmodule.angle(this_rot))
        average_angle += min(comparison_results)
    average_angle /= number_of_samples
    return average_angle


def average_relative_orientation(correct_rotations,
                                 recovered_rotations,
                                 symmetry_operations=((1., 0., 0., 0.), ),
                                 shallow_ewald=False):
    number_of_patterns = len(recovered_rotations)
    symmetry_operations = _numpy.array(symmetry_operations)
    if shallow_ewald:
        friedel_rotation = [
            _rotmodule.from_angle_and_dir(0, (0, 0, 1)),
            _rotmodule.from_angle_and_dir(_numpy.pi, (0, 0, 1))
        ]
    else:
        friedel_rotation = [
            _rotmodule.from_angle_and_dir(0, (0, 0, 1))
        ]

    all_symmetries = list(_itertools.product(symmetry_operations,
                                             friedel_rotation))

    relative_rotations = _numpy.zeros((recovered_rotations.shape[0], 4))
    symmetry_version = _numpy.zeros(recovered_rotations.shape[0],
                                    dtype=_numpy.int32)

    relative_rotations[0] = _rotmodule.multiply(
        correct_rotations[0],
        _rotmodule.inverse(recovered_rotations[0])
    )
    symmetry_version[0] = 0

    for index in range(1, number_of_patterns):
        relative_rot_sym = []
        for s, f in all_symmetries:
            this_relative_rot = _rotmodule.multiply(
                s,
                correct_rotations[index],
                _rotmodule.inverse(f),
                _rotmodule.inverse(recovered_rotations[index])
            )
            relative_rot_sym.append(this_relative_rot)

        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.fix_sign(this_relative_rot_sym)

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool_)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            # Try both positive and negative version of the rotation
            # since we don't know which one matches
            fit_quality_1 = _numpy.linalg.norm(relative_rotations[0] -
                                               this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rotations[0] +
                                               this_relative_rot)
            if fit_quality_1 < fit_quality_2:
                fit_quality[sym_index] = fit_quality_1
                flip[sym_index] = False
            else:
                fit_quality[sym_index] = fit_quality_2
                flip[sym_index] = True

        best_index = fit_quality.argmin()
        if flip[best_index]:
            relative_rotations[index, :] = -relative_rot_sym[best_index]
        else:
            relative_rotations[index, :] = relative_rot_sym[best_index]
        symmetry_version[index] = best_index

    average_rot = relative_rotations.mean(axis=0)
    average_rot = _rotmodule.normalize(average_rot)

    return average_rot


def absolute_orientation_error(correct_rotations,
                               recovered_rotations,
                               symmetry_operations=((1., 0., 0., 0.), ),
                               shallow_ewald=False,
                               return_individual_errors=False):
    number_of_patterns = len(recovered_rotations)
    symmetry_operations = _numpy.array(symmetry_operations)
    if shallow_ewald:
        friedel_rotation = [
            _rotmodule.from_angle_and_dir(0, (0, 0, 1)),
            _rotmodule.from_angle_and_dir(_numpy.pi, (0, 0, 1))
        ]
    else:
        friedel_rotation = [
            _rotmodule.from_angle_and_dir(0, (0, 0, 1))
        ]

    all_symmetries = list(_itertools.product(symmetry_operations,
                                             friedel_rotation))

    relative_rotations = _numpy.zeros((recovered_rotations.shape[0], 4))
    symmetry_version = _numpy.zeros(recovered_rotations.shape[0],
                                    dtype=_numpy.int32)

    relative_rotations[0] = _rotmodule.multiply(
        correct_rotations[0],
        _rotmodule.inverse(recovered_rotations[0])
    )
    symmetry_version[0] = 0

    for index in range(1, number_of_patterns):
        relative_rot_sym = []
        for s, f in all_symmetries:
            this_relative_rot = _rotmodule.multiply(
                s,
                correct_rotations[index],
                _rotmodule.inverse(f),
                _rotmodule.inverse(recovered_rotations[index])
            )
            relative_rot_sym.append(this_relative_rot)

        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.fix_sign(this_relative_rot_sym)

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool_)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            # Try both positive and negative version of the rotation
            # since we don't know which one matches
            fit_quality_1 = _numpy.linalg.norm(relative_rotations[0] -
                                               this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rotations[0] +
                                               this_relative_rot)
            if fit_quality_1 < fit_quality_2:
                fit_quality[sym_index] = fit_quality_1
                flip[sym_index] = False
            else:
                fit_quality[sym_index] = fit_quality_2
                flip[sym_index] = True

        best_index = fit_quality.argmin()
        if flip[best_index]:
            relative_rotations[index, :] = -relative_rot_sym[best_index]
        else:
            relative_rotations[index, :] = relative_rot_sym[best_index]
        symmetry_version[index] = best_index

    average_rot = relative_rotations.mean(axis=0)
    average_rot = _rotmodule.normalize(average_rot)

    diff_angles = _numpy.zeros(number_of_patterns)

    for index in range(number_of_patterns):
        # relative_angle(s, f)
        diff_angle = _rotmodule.relative_angle(
            _rotmodule.multiply(all_symmetries[symmetry_version[index]][0],
                                correct_rotations[index]),
            _rotmodule.multiply(average_rot,
                                recovered_rotations[index],
                                all_symmetries[symmetry_version[index]][1])
        )
        diff_angles[index] = diff_angle

    average_diff = diff_angles.mean()
    if return_individual_errors:
        return diff_angles
    else:
        return average_diff


get_absolute_orientation_error = refactor.new_to_old(
    absolute_orientation_error,
    "get_absolute_orientation_error")
get_relative_orientation_error = refactor.new_to_old(
    relative_orientation_error,
    "get_relative_orientation_error")
