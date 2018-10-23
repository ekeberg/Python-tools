import numpy as _numpy
from . import rotations as _rotmodule

def relative(rot1, rot2):
    return _rotmodule.quaternion_multiply(_rotmodule.quaternion_inverse(rot1), rot2)

def get_angle(rot1, rot2):
    w = relative(rot1, rot2)[0]
    if w > 1:
        print("w = {0}".format(w))
        w = 1.
    if w < -1:
        print("w = {0}".format(w))
        w = -1.
    diff_angle = 2.*_numpy.arccos(w)
    abs_diff_angle = min(abs(diff_angle), abs(diff_angle-2.*_numpy.pi))
    return abs_diff_angle

def get_fit_quality(this_rot, reference_rot):
    return min([_numpy.linalg.norm(reference_rot - this_rot),
                _numpy.linalg.norm(reference_rot + this_rot)])

def average_relative_orientation(rotations_1, rotations_2, symmetry_operations):
    number_of_patterns = len(rotations_2)
    average_angular_diff = 0.
    count = 0.

    relative_rotations = _numpy.zeros((rotations_2.shape[0], 4))
    symmetry_version = _numpy.zeros(rotations_2.shape[0], dtype=_numpy.int32)
    for index in range(number_of_patterns):
        relative_rot = _rotmodule.quaternion_multiply(rotations_2[index],
                                                     _rotmodule.quaternion_inverse(rotations_1[index]))
        _rotmodule.quaternion_fix_sign(relative_rot)
        relative_rot_sym = [_rotmodule.quaternion_multiply(relative_rot, this_symmetry_operation)
                            for this_symmetry_operation in symmetry_operations]
        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.quaternion_fix_sign(this_relative_rot_sym)

        reference_rot = _rotmodule.random_quaternion()

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool8)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            fit_quality_1 = _numpy.linalg.norm(relative_rot[0] - this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rot[0] + this_relative_rot)
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
    average_rot = _rotmodule.quaternion_normalize(average_rot)
    return average_rot

def get_absolute_orientation_error(correct_rotations, recovered_rotations,
                                   symmetry_operations=((1., 0., 0., 0.), ),
                                   return_individual_errors=False):
    number_of_patterns = len(recovered_rotations)
    average_angular_diff = 0.
    count = 0.

    relative_rotations = _numpy.zeros((recovered_rotations.shape[0], 4))
    symmetry_version = _numpy.zeros(recovered_rotations.shape[0], dtype=_numpy.int32)
    for index in range(number_of_patterns):
        relative_rot = _rotmodule.quaternion_multiply(recovered_rotations[index],
                                                     _rotmodule.quaternion_inverse(correct_rotations[index]))
        _rotmodule.quaternion_fix_sign(relative_rot)
        relative_rot_sym = [_rotmodule.quaternion_multiply(relative_rot, this_symmetry_operation)
                            for this_symmetry_operation in symmetry_operations]
        for this_relative_rot_sym in relative_rot_sym:
            _rotmodule.quaternion_fix_sign(this_relative_rot_sym)

        reference_rot = _rotmodule.random_quaternion()

        fit_quality = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.float64)
        flip = _numpy.zeros(len(relative_rot_sym), dtype=_numpy.bool8)
        for sym_index, this_relative_rot in enumerate(relative_rot_sym):
            fit_quality_1 = _numpy.linalg.norm(relative_rot[0] - this_relative_rot)
            fit_quality_2 = _numpy.linalg.norm(relative_rot[0] + this_relative_rot)
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
    average_rot = _rotmodule.quaternion_normalize(average_rot)

    diff_angles = _numpy.zeros(number_of_patterns)
    
    for index in range(number_of_patterns):
        adjusted_rotation = _rotmodule.quaternion_multiply(_rotmodule.quaternion_multiply(symmetry_operations[symmetry_version[index]], _rotmodule.quaternion_inverse(average_rot)), recovered_rotations[index])
        relative_rotation = _rotmodule.quaternion_multiply(_rotmodule.quaternion_inverse(adjusted_rotation),
                                                          correct_rotations[index])
        diff_angle = get_angle(adjusted_rotation, correct_rotations[index])
        diff_angles[index] = diff_angle
        
    average_diff = diff_angles.mean()
    if return_individual_errors:
        return diff_angles
    else:
        return average_diff

def get_relative_orientation_error(correct_rotations, recovered_rotations, symmetry_operations):
    number_of_samples = 1000
    number_of_rotations = len(recovered_rotations)
    average_angle = 0.
    for _ in range(number_of_samples):
        index_1 = _numpy.random.randint(number_of_rotations)
        index_2 = _numpy.random.randint(number_of_rotations)
        if index_1 == index_2:
            index_2 = (index_1 + 1) % number_of_rotations

        correct_relative = [relative(correct_rotations[index_1],
                                     _rotmodule.quaternion_multiply(this_symmetry_operation,
                                                                   correct_rotations[index_2]))
                            for this_symmetry_operation in symmetry_operations]
        for this_correct_relative in correct_relative:
            _rotmodule.quaternion_fix_sign(this_correct_relative)

        recovered_relative = relative(recovered_rotations[index_1],
                                      recovered_rotations[index_2])
        _rotmodule.quaternion_fix_sign(recovered_relative)

        angle = [get_angle(this_correct_relative, recovered_relative)
                 for this_correct_relative in correct_relative]
        diff_angle = min(angle)
        average_angle += diff_angle
            
    average_angle /= number_of_samples
    return average_angle
