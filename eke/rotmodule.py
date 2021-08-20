"""A set of tools to handle quaternions and other functions related to
spatial rotation."""
import numpy as _numpy

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
    raise NotImplementedError("Convert to matrix and then to euler angle instead.")
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
    return _numpy.array([[quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2,
                          2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3],
                          2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2],],
                         [2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3],
                          quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2,
                          2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1]],
                         [2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2],
                          2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1],
                          quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2]])

def quaternion_to_matrix_bw(quat):
    """Dummy docstring"""
    return _numpy.array([[quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2,
                          2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1],
                          2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2]],
                         [2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1],
                          quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2,
                          2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3]],
                         [2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2],
                          2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3],
                          quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2]])

def inverse(quat_in):
    """Return the inverse of the quaternion. Input is unchanged."""
    quat_out = _numpy.zeros(quat_in.shape)
    quat_out[..., 0] = quat_in[..., 0]
    quat_out[..., 1:] = -quat_in[..., 1:]
    return quat_out.squeeze()

def normalize(quat):
    """Normalize the quaternion and return the same object. (input quaternion is changed)"""
    if len(quat.shape) == 1:
        quat = quat.reshape((1, 4))
    norm = _numpy.linalg.norm(quat, axis=1)
    fix_sign(quat)
    return (quat/norm[:, _numpy.newaxis]).squeeze()

def fix_sign(quat):
    shape = (1, 4) if len(quat.shape) == 1 else quat.shape
    quat_array = quat.reshape(shape)
    under_consideration = _numpy.ones(quat_array.shape[0], dtype="bool8")
    for index in range(4):
        do_flip = under_consideration & (quat_array[..., index] < 0)
        quat_array[do_flip, :] = -quat_array[do_flip, :]
        under_consideration &= quat_array[..., index] == 0.
            
def _multiply_single(quat_1, quat_2):
    """Return the product of quat_1 and quat_2"""
    quat_1 = _numpy.array(quat_1)
    quat_2 = _numpy.array(quat_2)
    if len(quat_1.shape) == 0:
        quat_1 = quat_1.reshape((1, 4))
        quat_2 = quat_2.reshape((1, 4))
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
    w = quat[0]
    if w > 1:
        w = 1.
    if w < -1:
        w = -1.
    diff_angle = 2.*_numpy.arccos(w)
    abs_diff_angle = min(abs(diff_angle), abs(diff_angle-2.*_numpy.pi))
    return abs_diff_angle

def relative_angle(rot1, rot2):
    """Angle of the relative orientation from rot1 to rot2"""
    return angle(relative(rot1, rot2))

# def rotate(quat, point):
#     """Rotate a point by the quaternion"""
#     rotation_matrix = quaternion_to_matrix(quat)
#     return _numpy.squeeze(_numpy.array(
#         rotation_matrix*_numpy.transpose(_numpy.matrix(point))))

# def rotate(quat, z_coordinates, y_coordinates, x_coordinates):
#     """Rotate coordinate vectors x, y and z by the rotation defined by the quaternion."""
#     rotation_matrix = quaternion_to_matrix(quat)
#     out_matrix = rotation_matrix*_numpy.matrix([z_coordinates, y_coordinates, x_coordinates])
#     out_array = _numpy.array(out_matrix)
#     return out_array[0], out_array[1], out_array[2]

def rotate(quat, coordinates):
    rotation_matrix = quaternion_to_matrix(quat)
    return rotation_matrix.dot(coordinates.T).T

def rotate_array_bw(quat, x_coordinates, y_coordinates, z_coordinates):
    """Like rotate_array but with the coordintes index backwords. Do not use."""
    rotation_matrix = quaternion_to_matrix_bw(quat)
    out_array = rotation_matrix.dot(numpy.array([x_coordinates, y_coordinates, z_coordinates]))
    return out_array[0], out_array[1], out_array[2]

def read_quaternion_list(filename):
    """Read an hdf5 file with quaternions. Quaternions are stored separately in
    fields named '1', '2', .... A field 'number_of_rotations' define the number of
    such fields."""
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
            raise ValueError("%d rotations does not correspond to any n" % rots)
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
    if order == "xyz":
        euler = _numpy.array([_numpy.arctan2(-mat[1, 2], mat[2, 2]),
                             _numpy.arctan2(mat[0, 2], _numpy.sqrt(1.-mat[0, 2]**2)),
                             _numpy.arctan2(-mat[0, 1], mat[0, 0])])
    elif order == "xzy":
        euler = _numpy.array([_numpy.arctan2(mat[2, 1], mat[1, 1]),
                             _numpy.arctan2(-mat[0, 1], _numpy.sqrt(1.-mat[0, 1]**2)),
                             _numpy.arctan2(mat[0, 2], mat[0, 0])])
    elif order == "xyx":
        euler = _numpy.array([_numpy.arctan2(mat[1, 0], -mat[2, 0]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[0, 0]**2), mat[0, 0]),
                             _numpy.arctan2(mat[0, 1], mat[0, 2])])
    elif order == "xzx":
        euler = _numpy.array([_numpy.arctan2(mat[2, 0], mat[1, 0]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[0, 0]**2), mat[0, 0]),
                             _numpy.arctan2(mat[0, 2], -mat[0, 1])])
    elif order == "yxz":
        euler = _numpy.array([_numpy.arctan2(mat[2, 0], mat[2, 2]),
                             _numpy.arctan2(-mat[1, 2], _numpy.sqrt(1.-mat[1, 2]**2)),
                             _numpy.arctan2(mat[1, 0], -mat[1, 1])])
    elif order == "yzx":
        euler = _numpy.array([_numpy.arctan2(-mat[2, 0], mat[0, 0]),
                             _numpy.arctan2(mat[1, 0], _numpy.sqrt(1.-mat[1, 0]**2)),
                             _numpy.arctan2(-mat[1, 2], mat[1, 1])])
    elif order == "yxy":
        euler = _numpy.array([_numpy.arctan2(mat[0, 1], mat[2, 1]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[1, 1]**2), mat[1, 1]),
                             _numpy.arctan2(-mat[1, 0], -mat[1, 0])])
    elif order == "yzy":
        euler = _numpy.array([_numpy.arctan2(mat[2, 1], -mat[0, 1]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[1, 1]**2), mat[1, 1]),
                             _numpy.arctan2(mat[1, 2], mat[1, 0])])
    elif order == "zxy":
        euler = _numpy.array([_numpy.arctan2(-mat[0, 1], mat[1, 1]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[2, 1]**2), mat[2, 1]),
                             _numpy.arctan2(mat[2, 0], mat[2, 2])])
    elif order == "zyx":
        euler = _numpy.array([_numpy.arctan2(mat[1, 0], mat[0, 0]),
                             _numpy.arctan2(-mat[2, 0], _numpy.sqrt(1.-mat[2, 0]**2)),
                             _numpy.arctan2(mat[2, 1], mat[2, 2])])
    elif order == "zxz":
        euler = _numpy.array([_numpy.arctan2(mat[0, 2], -mat[1, 2]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[2, 2]**2), mat[2, 2]),
                             _numpy.arctan2(mat[2, 0], mat[2, 1])])
    elif order == "zyz":
        euler = _numpy.array([_numpy.arctan2(mat[1, 2], mat[0, 2]),
                             _numpy.arctan2(_numpy.sqrt(1.-mat[2, 2]**2), mat[2, 2]),
                             _numpy.arctan2(mat[2, 1], -mat[2, 0])])
    else:
        raise ValueError("unrecognized order: {0}".format(order))
        
    return euler

from . import refactor
rotate_array = refactor.new_to_old(rotate, "rotate_array")

random_quaternion = refactor.new_to_old(random, "random_quaternion")
quaternion_from_dir_and_angle = refactor.new_to_old(
    from_angle_and_dir, "quaternion_from_dir_and_angle")
quaternion_inverse = refactor.new_to_old(inverse, "quaternion_inverse")
quaternion_normalize = refactor.new_to_old(normalize, "quaternion_normalize")
quaternion_fix_sign = refactor.new_to_old(fix_sign, "quaternion_fix_sign")
quaternion_multiply = refactor.new_to_old(multiply, "quaternion_multiply")
quaternion_relative = refactor.new_to_old(relative, "quaternion_relative")
