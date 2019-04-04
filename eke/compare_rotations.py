import numpy as _numpy
from . import rotmodule as _rotmodule

def relative_angle(rot1, rot2):
    """Angle of the relative orientation from rot1 to rot2"""
    w = _rotmodule.relative(rot1, rot2)[0]
    if w > 1:
        print("w = {0}".format(w))
        w = 1.
    if w < -1:
        print("w = {0}".format(w))
        w = -1.
    diff_angle = 2.*_numpy.arccos(w)
    abs_diff_angle = min(abs(diff_angle), abs(diff_angle-2.*_numpy.pi))
    return abs_diff_angle

def average_relative_orientation(
        rotations_1, rotations_2, symmetry_operations=((1., 0., 0., 0.), )):
    number_of_patterns = len(rotations_2)
    average_angular_diff = 0.
    count = 0.

    relative_rotations = _numpy.zeros((rotations_2.shape[0], 4))
    symmetry_version = _numpy.zeros(rotations_2.shape[0], dtype=_numpy.int32)
    for index in range(number_of_patterns):
        relative_rot = _rotmodule.multiply(
            rotations_2[index], _rotmodule.inverse(rotations_1[index]))
        _rotmodule.fix_sign(relative_rot)
        relative_rot_sym = [_rotmodule.multiply(relative_rot,
                                                this_symmetry_operation)
                            for this_symmetry_operation in symmetry_operations]
        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.fix_sign(this_relative_rot_sym)

        reference_rot = _rotmodule.random()

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool8)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            fit_quality_1 = _numpy.linalg.norm(relative_rot[0] -
                                               this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rot[0] +
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

def absolute_orientation_error(
        correct_rotations, recovered_rotations,
        symmetry_operations=((1., 0., 0., 0.), )):
    number_of_patterns = len(recovered_rotations)
    average_angular_diff = 0.
    count = 0.
    symmetry_operations = _numpy.array(symmetry_operations)

    relative_rotations = _numpy.zeros((recovered_rotations.shape[0], 4))
    symmetry_version = _numpy.zeros(recovered_rotations.shape[0], dtype=_numpy.int32)

    relative_rotations[0] = _rotmodule.relative(recovered_rotations[0], correct_rotations[0])
    symmetry_version[0] = 0
    for index in range(1, number_of_patterns):
        # relative_rot_sym = [_rotmodule.relative(_rotmodule.multiply(_rotmodule.inverse(this_symmetry_operation),
        #                                                             recovered_rotations[index]),
        #                                         correct_rotations[index])
        #                     for this_symmetry_operation in symmetry_operations]
        relative_rot_sym = [_rotmodule.multiply(_rotmodule.multiply(recovered_rotations[index],
                                                                    _rotmodule.inverse(correct_rotations[index])),
                                                _rotmodule.inverse(this_symmetry_operation))
                            for this_symmetry_operation in symmetry_operations]
        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.fix_sign(this_relative_rot_sym)

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool8)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            # Try both positive and negative version of the rotation
            # since we don't know which one matches
            fit_quality_1 = _numpy.linalg.norm(relative_rotations[0] - this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rotations[0] + this_relative_rot)
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

    # print(relative_rotations)
    average_rot = relative_rotations.mean(axis=0)
    # print(_numpy.sqrt((average_rot**2).sum()))
    # print(symmetry_version)
    # import ipdb; ipdb.set_trace()
    average_rot = _rotmodule.normalize(average_rot)

    # average = inv(recovered * symmetry) * correct
    # average = inv(symmetry) * inv(recovered) * correct
    # inv(correct) = inv(average) * inv(symmetry) * inv(recovered)
    # correct = recovered * symmetry * average

    average_diff = 0.
    for index in range(number_of_patterns):
        adjusted_rotation = _rotmodule.multiply(_rotmodule.multiply(_rotmodule.inverse(symmetry_operations[symmetry_version[index]]), _rotmodule.inverse(average_rot)), recovered_rotations[index])
        diff_angle = relative_angle(correct_rotations[index], adjusted_rotation)
        average_diff += diff_angle
    average_diff /= number_of_patterns
    return average_diff

def relative_orientation_error(
        correct_rotations, recovered_rotations,
        symmetry_operations=((1., 0., 0., 0.), )):
    symmetry_operations = _numpy.array(symmetry_operations)
    number_of_samples = 20 #3000
    number_of_rotations = len(recovered_rotations)
    average_angle = 0.
    for _ in range(number_of_samples):
        index_1 = _numpy.random.randint(number_of_rotations)
        index_2 = _numpy.random.randint(number_of_rotations)
        if index_1 == index_2:
            index_2 = (index_1 + 1) % number_of_rotations

        recovered_relative = _rotmodule.relative(recovered_rotations[index_1],
                                                 recovered_rotations[index_2])
        _rotmodule.fix_sign(recovered_relative)
        correct_relative = [_rotmodule.relative(correct_rotations[index_1], _rotmodule.multiply(this_symmetry_operation, correct_rotations[index_2]))
                            for this_symmetry_operation in symmetry_operations]
        for this_correct_relative in correct_relative:
            _rotmodule.fix_sign(this_correct_relative)
        
        angle = [relative_angle(this_correct_relative, recovered_relative)
                 for this_correct_relative in correct_relative]
        diff_angle = min(angle)
        average_angle += diff_angle
    average_angle /= number_of_samples
    return average_angle



from . import refactor
get_absolute_orientation_error = refactor.new_to_old(
    absolute_orientation_error,
    "get_absolute_orientation_error")
get_relative_orientation_error = refactor.new_to_old(
    relative_orientation_error,
    "get_relative_orientation_error")
