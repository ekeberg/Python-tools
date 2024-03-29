"""A set of tools to handle quaternions and other functions related to
spatial rotation."""
import numpy as _numpy
from . import refactor


def random(number_of_quaternions=1, fix_sign=True):
    random_base = _numpy.random.random((number_of_quaternions, 3))
    quats = _numpy.empty((number_of_quaternions, 4))
    quats[:, 0] = (_numpy.sqrt(1.-random_base[:, 0]) *
                   _numpy.sin(2.*_numpy.pi*random_base[:, 1]))
    quats[:, 1] = (_numpy.sqrt(1.-random_base[:, 0]) *
                   _numpy.cos(2.*_numpy.pi*random_base[:, 1]))
    quats[:, 2] = (_numpy.sqrt(random_base[:, 0]) *
                   _numpy.sin(2.*_numpy.pi*random_base[:, 2]))
    quats[:, 3] = (_numpy.sqrt(random_base[:, 0]) *
                   _numpy.cos(2.*_numpy.pi*random_base[:, 2]))
    if fix_sign:
        globals()["fix_sign"](quats)
    return quats.squeeze()


def from_angle_and_dir(angle, direction):
    """The quaternion corresponding to a rotations of angle (rad) around the
    axis defined by direction. The rotation follows the right-hand rule."""
    angle = _numpy.mod(angle, 2.*_numpy.pi)
    quaternion = _numpy.zeros(4)
    quaternion[0] = _numpy.cos(angle/2.)
    normalization = _numpy.linalg.norm(direction)
    modulation = _numpy.sqrt(1.-quaternion[0]**2)
    quaternion[1:] = _numpy.array(direction)/normalization*modulation
    fix_sign(quaternion)
    return quaternion


def quaternion_to_euler_angle(quat):
    """Generate euler angles from the quaternion. The last angle
    corresponds to in-plane rotation."""
    raise NotImplementedError("Convert to matrix and then to euler angle "
                              "instead.")
    euler = _numpy.zeros(3)
    euler[0] = _numpy.arctan2(2.0*(quat[0]*quat[1] + quat[2]*quat[3]),
                              1.0 - 2.0*(quat[1]**2 + quat[2]**2))
    arcsin_argument = 2.0*(quat[0]*quat[2] - quat[1]*quat[3])
    if arcsin_argument > 1.0:
        arcsin_argument = 1.0
    if arcsin_argument < -1.0:
        arcsin_argument = -1.0
    euler[1] = _numpy.arcsin(arcsin_argument)
    euler[2] = _numpy.arctan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]),
                              1.0 - 2.0*(quat[2]**2 + quat[3]**2))
    return euler


def quaternion_to_matrix(quat):
    """Dummy docstring"""
    quat = _numpy.array(quat)
    if len(quat.shape) < 2:
        matrix = _numpy.zeros((3, 3), dtype=quat.dtype)
    else:
        matrix = _numpy.zeros((len(quat), 3, 3), dtype=quat.dtype)

    matrix[..., 0, 0] = (quat[..., 0]**2 + quat[..., 1]**2
                         - quat[..., 2]**2 - quat[..., 3]**2)
    matrix[..., 0, 1] = (2 * quat[..., 1] * quat[..., 2]
                         - 2 * quat[..., 0] * quat[..., 3])
    matrix[..., 0, 2] = (2 * quat[..., 1] * quat[..., 3]
                         + 2 * quat[..., 0] * quat[..., 2])
    matrix[..., 1, 0] = (2 * quat[..., 1] * quat[..., 2]
                         + 2 * quat[..., 0] * quat[..., 3])
    matrix[..., 1, 1] = (quat[..., 0]**2 - quat[..., 1]**2
                         + quat[..., 2]**2 - quat[..., 3]**2)
    matrix[..., 1, 2] = (2 * quat[..., 2] * quat[..., 3]
                         - 2 * quat[..., 0] * quat[..., 1])
    matrix[..., 2, 0] = (2 * quat[..., 1] * quat[..., 3]
                         - 2 * quat[..., 0] * quat[..., 2])
    matrix[..., 2, 1] = (2 * quat[..., 2] * quat[..., 3]
                         + 2 * quat[..., 0] * quat[..., 1])
    matrix[..., 2, 2] = (quat[..., 0]**2 - quat[..., 1]**2
                         - quat[..., 2]**2 + quat[..., 3]**2)
    return matrix


def quaternion_to_matrix_bw(quat):
    """Dummy docstring"""
    if len(quat.shape) < 2:
        matrix = _numpy.zeros((3, 3), dtype=quat.dtype)
    else:
        matrix = _numpy.zeros((len(quat), 3, 3), dtype=quat.dtype)

    matrix[..., 0, 0] = quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2
    matrix[..., 0, 1] = 2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1]
    matrix[..., 0, 2] = 2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2]
    matrix[..., 1, 0] = 2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1]
    matrix[..., 1, 1] = quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2
    matrix[..., 1, 2] = 2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3]
    matrix[..., 2, 0] = 2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2]
    matrix[..., 2, 1] = 2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3]
    matrix[..., 2, 2] = quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2
    return matrix


def inverse(quat_in):
    """Return the inverse of the quaternion. Input is unchanged."""
    quat_out = _numpy.zeros(quat_in.shape)
    quat_out[..., 0] = quat_in[..., 0]
    quat_out[..., 1:] = -quat_in[..., 1:]
    return quat_out.squeeze()


def normalize(quat):
    """Normalize the quaternion and return the same object. (input
    quaternion is changed)"""
    if len(quat.shape) == 1:
        quat = quat.reshape((1, 4))
    norm = _numpy.linalg.norm(quat, axis=1)
    fix_sign(quat)
    return (quat/norm[:, _numpy.newaxis]).squeeze()


def fix_sign(quat):
    shape = (1, 4) if len(quat.shape) == 1 else quat.shape
    quat_array = quat.reshape(shape)
    under_consideration = _numpy.ones(quat_array.shape[0], dtype="bool")
    for index in range(4):
        do_flip = under_consideration & (quat_array[..., index] < 0)
        quat_array[do_flip, :] = -quat_array[do_flip, :]
        under_consideration &= quat_array[..., index] == 0.


def _multiply_single(quat_1, quat_2):
    """Return the product of quat_1 and quat_2"""
    quat_1 = _numpy.array(quat_1)
    quat_2 = _numpy.array(quat_2)
    if (len(quat_1.shape) == 2 and len(quat_2.shape) == 1):
        quat_2 = quat_2.reshape((1, 4))
        quat_out = _numpy.zeros(quat_1.shape)
    if (len(quat_2.shape) == 2 and len(quat_1.shape) == 1):
        quat_1 = quat_1.reshape((1, 4))
        quat_out = _numpy.zeros(quat_2.shape)
    else:
        quat_out = _numpy.zeros(quat_1.shape)

    quat_out[..., 0] = (quat_1[..., 0]*quat_2[..., 0] -
                        quat_1[..., 1]*quat_2[..., 1] -
                        quat_1[..., 2]*quat_2[..., 2] -
                        quat_1[..., 3]*quat_2[..., 3])
    quat_out[..., 1] = (quat_1[..., 0]*quat_2[..., 1] +
                        quat_1[..., 1]*quat_2[..., 0] +
                        quat_1[..., 2]*quat_2[..., 3] -
                        quat_1[..., 3]*quat_2[..., 2])
    quat_out[..., 2] = (quat_1[..., 0]*quat_2[..., 2] -
                        quat_1[..., 1]*quat_2[..., 3] +
                        quat_1[..., 2]*quat_2[..., 0] +
                        quat_1[..., 3]*quat_2[..., 1])
    quat_out[..., 3] = (quat_1[..., 0]*quat_2[..., 3] +
                        quat_1[..., 1]*quat_2[..., 2] -
                        quat_1[..., 2]*quat_2[..., 1] +
                        quat_1[..., 3]*quat_2[..., 0])
    return quat_out.squeeze()


def multiply(*list_of_quaternions):
    """Return the product of all the provided quaternions"""
    if len(list_of_quaternions) < 1:
        raise ValueError("Must provide at least one quaternion to multiply")
    result = list_of_quaternions[0]
    for this_rot in list_of_quaternions[1:]:
        result = _multiply_single(result, this_rot)
    return result


def relative(quat_1, quat_2):
    return multiply(inverse(quat_1), quat_2)


def angle(quat):
    """Angle of the rotation"""
    quat = _numpy.array(quat)
    if len(quat.shape) == 1:
        quat = quat.reshape((1, 4))
    w = quat[:, 0]
    w[w > 1] = 1
    w[w < -1] = -1
    diff_angle = 2.*_numpy.arccos(w)
    abs_diff_angle = _numpy.minimum(abs(diff_angle),
                                    abs(diff_angle-2.*_numpy.pi))
    return abs_diff_angle.squeeze()


def relative_angle(rot1, rot2):
    """Angle of the relative orientation from rot1 to rot2"""
    return angle(relative(rot1, rot2))


def rotate(quat, vec):
    """Rotate the vector by the quaternion. Quat is expected to have shape
    (..., 4) and vec is expected to have shape (..., 3)."""
    rot_matrix = quaternion_to_matrix(quat)
    
    # What is the shape of the input, in addition to the
    # minimal quaternion and vector (3D) shapes?
    # This expects the extra dims to be at the beginning
    quat_extra_dim = quat.shape[:-1]
    vec_extra_dim = vec.shape[:-1]

    # Transpose the vector so the first dimension is the 3D vector
    # so that it's ready for matrix multiplication
    vec_transpose = vec.transpose((-1, ) + tuple(range(len(vec.shape) - 1)))
    result = rot_matrix @ vec_transpose

    # Transpose the result back so that the 3D vector is the last dimension
    reorder = list(range(len(quat_extra_dim)))
    reorder += list(range(len(quat_extra_dim) + 1, len(vec_extra_dim) + len(quat_extra_dim) + 1))
    reorder.append(len(quat_extra_dim))

    result = result.transpose(reorder)
    return result


def rotate_array_bw(quat, x_coordinates, y_coordinates, z_coordinates):
    """Like rotate_array but with the coordintes index backwords. Do not
    use."""
    rotation_matrix = quaternion_to_matrix_bw(quat)
    out_array = rotation_matrix.dot(_numpy.array([x_coordinates,
                                                  y_coordinates,
                                                  z_coordinates]))
    return out_array[0], out_array[1], out_array[2]


def read_quaternion_list(filename):
    """Read an hdf5 file with quaternions. Quaternions are stored
    separately in fields named '1', '2', .... A field
    'number_of_rotations' define the number of such fields.
    """
    import h5py
    with h5py.File(filename) as file_handle:
        number_of_rotations = file_handle['number_of_rotations'][...]
        quaternions = _numpy.zeros((number_of_rotations, 4))
        weights = _numpy.zeros(number_of_rotations)
        for i in range(number_of_rotations):
            quaternion_and_weight = file_handle[str(i)][...]
            quaternions[i] = quaternion_and_weight[:4]
            weights[i] = quaternion_and_weight[4]
    return quaternions, weights


def n_to_rots(sampling_n):
    """The number of rotations when the sampling parameter n is used"""
    return 10*(sampling_n+5*sampling_n**3)


def n_to_angle(sampling_n):
    """The largest angle separating two adjacant orientations when the
    sampling parameter n is used."""
    tau = (1.0 + _numpy.sqrt(5.0))/2.0
    return 4./sampling_n/tau**3


def rots_to_n(rots):
    """The sampling parameter n corresponding to a certain
    number of rotations."""
    sampling_n = 1
    while True:
        this_rots = n_to_rots(sampling_n)
        if this_rots == rots:
            return sampling_n
        if this_rots > rots:
            raise ValueError(f"{rots} rotations does not correspond to any n")
        sampling_n += 1


def quaternion_to_angle(quat):
    """The angle by which this transformation rotates"""
    return 2.*_numpy.arccos(quat[0])


def quaternion_to_axis(quat):
    """The axis around which this rotation rotates"""
    return quat[1:]/_numpy.linalg.norm(quat[1:])


def matrix_to_euler_angle(mat, order="zxz"):
    """This function is based on the following NASA paper:
    http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf"""

    if len(mat.shape) < 3:
        euler = _numpy.zeros(3, dtype=mat.dtype)
    else:
        euler = _numpy.zeros((len(mat), 3), dtype=mat.dtype)

    if order == "xyz":
        euler[..., 0] = _numpy.arctan2(-mat[..., 1, 2], mat[..., 2, 2])
        euler[..., 1] = _numpy.arctan2(mat[..., 0, 2],
                                       _numpy.sqrt(1.-mat[..., 0, 2]**2))
        euler[..., 2] = _numpy.arctan2(-mat[..., 0, 1], mat[..., 0, 0])
    elif order == "xzy":
        euler[..., 0] = _numpy.arctan2(mat[..., 2, 1], mat[..., 1, 1])
        euler[..., 1] = _numpy.arctan2(-mat[..., 0, 1],
                                       _numpy.sqrt(1.-mat[..., 0, 1]**2))
        euler[..., 2] = _numpy.arctan2(mat[..., 0, 2], mat[..., 0, 0])
    elif order == "xyx":
        euler[..., 0] = _numpy.arctan2(mat[..., 1, 0], -mat[..., 2, 0])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 0, 0]**2),
                                       mat[..., 0, 0])
        euler[..., 2] = _numpy.arctan2(mat[..., 0, 1], mat[..., 0, 2])
    elif order == "xzx":
        euler[..., 0] = _numpy.arctan2(mat[..., 2, 0], mat[..., 1, 0])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 0, 0]**2),
                                       mat[..., 0, 0])
        euler[..., 2] = _numpy.arctan2(mat[..., 0, 2], -mat[..., 0, 1])
    elif order == "yxz":
        euler[..., 0] = _numpy.arctan2(mat[..., 2, 0], mat[..., 2, 2])
        euler[..., 1] = _numpy.arctan2(-mat[..., 1, 2],
                                       _numpy.sqrt(1.-mat[..., 1, 2]**2))
        euler[..., 2] = _numpy.arctan2(mat[..., 1, 0], -mat[..., 1, 1])
    elif order == "yzx":
        euler[..., 0] = _numpy.arctan2(-mat[..., 2, 0], mat[..., 0, 0])
        euler[..., 1] = _numpy.arctan2(mat[..., 1, 0],
                                       _numpy.sqrt(1.-mat[..., 1, 0]**2))
        euler[..., 2] = _numpy.arctan2(-mat[..., 1, 2], mat[..., 1, 1])
    elif order == "yxy":
        euler[..., 0] = _numpy.arctan2(mat[..., 0, 1], mat[..., 2, 1])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 1, 1]**2),
                                       mat[..., 1, 1])
        euler[..., 2] = _numpy.arctan2(-mat[..., 1, 0], -mat[..., 1, 0])
    elif order == "yzy":
        euler[..., 0] = _numpy.arctan2(mat[..., 2, 1], -mat[..., 0, 1])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 1, 1]**2),
                                       mat[..., 1, 1])
        euler[..., 2] = _numpy.arctan2(mat[..., 1, 2], mat[..., 1, 0])
    elif order == "zxy":
        euler[..., 0] = _numpy.arctan2(-mat[..., 0, 1], mat[..., 1, 1])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 2, 1]**2),
                                       mat[..., 2, 1])
        euler[..., 2] = _numpy.arctan2(mat[..., 2, 0], mat[..., 2, 2])
    elif order == "zyx":
        euler[..., 0] = _numpy.arctan2(mat[..., 1, 0], mat[..., 0, 0])
        euler[..., 1] = _numpy.arctan2(-mat[..., 2, 0],
                                       _numpy.sqrt(1.-mat[..., 2, 0]**2))
        euler[..., 2] = _numpy.arctan2(mat[..., 2, 1], mat[..., 2, 2])
    elif order == "zxz":
        euler[..., 0] = _numpy.arctan2(mat[..., 0, 2], -mat[..., 1, 2])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 2, 2]**2),
                                       mat[..., 2, 2])
        euler[..., 2] = _numpy.arctan2(mat[..., 2, 0], mat[..., 2, 1])
    elif order == "zyz":
        euler[..., 0] = _numpy.arctan2(mat[..., 1, 2], mat[..., 0, 2])
        euler[..., 1] = _numpy.arctan2(_numpy.sqrt(1.-mat[..., 2, 2]**2),
                                       mat[..., 2, 2])
        euler[..., 2] = _numpy.arctan2(mat[..., 2, 1], -mat[..., 2, 0])
    else:
        raise ValueError("unrecognized order: {0}".format(order))

    return euler


rotate_array = refactor.new_to_old(rotate, "rotate_array")


random_quaternion = refactor.new_to_old(random, "random_quaternion")
quaternion_from_dir_and_angle = refactor.new_to_old(
    from_angle_and_dir, "quaternion_from_dir_and_angle")
quaternion_inverse = refactor.new_to_old(inverse, "quaternion_inverse")
quaternion_normalize = refactor.new_to_old(normalize, "quaternion_normalize")
quaternion_fix_sign = refactor.new_to_old(fix_sign, "quaternion_fix_sign")
quaternion_multiply = refactor.new_to_old(multiply, "quaternion_multiply")
quaternion_relative = refactor.new_to_old(relative, "quaternion_relative")
