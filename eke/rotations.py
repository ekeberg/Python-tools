"""A set of tools to handle quaternions and other functions related to
spatial rotation."""
import numpy as _numpy

def random_quaternion(number_of_quaternions=1):
    random_base = _numpy.random.random((number_of_quaternions, 3))
    quats = _numpy.empty((number_of_quaternions, 4))
    quats[:, 0] = _numpy.sqrt(1.-random_base[:, 0]) * _numpy.sin(2.*_numpy.pi*random_base[:, 1])
    quats[:, 1] = _numpy.sqrt(1.-random_base[:, 0]) * _numpy.cos(2.*_numpy.pi*random_base[:, 1])
    quats[:, 2] = _numpy.sqrt(random_base[:, 0]) * _numpy.sin(2.*_numpy.pi*random_base[:, 2])
    quats[:, 3] = _numpy.sqrt(random_base[:, 0]) * _numpy.cos(2.*_numpy.pi*random_base[:, 2])
    return quats.squeeze()

def quaternion_from_dir_and_angle(angle, direction):
    """The quaternion corresponding to a rotations of angle (rad) around the
    axis defined by direction. The rotation follows the right-hand rule."""
    quaternion = _numpy.zeros(4)
    #normalized_dir = _numpy.array(dir)/norm(dir)
    quaternion[0] = _numpy.cos(angle/2.)
    quaternion[1:] = _numpy.array(direction)/_numpy.linalg.norm(direction)*_numpy.sqrt(1.-quaternion[0]**2)
    return quaternion

def quaternion_to_euler_angle(quat):
    """Generate euler angles from the quaternion. The last angle corresponds to in-plane rotation."""
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
    return _numpy.matrix([[quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2,
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
    return _numpy.matrix([[quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2,
                          2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1],
                          2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2]],
                         [2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1],
                          quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2,
                          2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3]],
                         [2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2],
                          2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3],
                          quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2]])

def quaternion_inverse(quat_in):
    """Return the inverse of the quaternion. Input is unchanged."""
    quat_out = _numpy.zeros(4)
    quat_out[0] = quat_in[0]
    quat_out[1:] = -quat_in[1:]
    return quat_out

def quaternion_normalize(quat):
    """Normalize the quaternion and return the same object. (input quaternion is changed)"""
    norm = _numpy.linalg.norm(quat)
    return quat/norm

def quaternion_multiply(quat_1, quat_2):
    """Return the product of quat_1 and quat_2"""
    quat_out = _numpy.zeros(4)
    quat_out[0] = quat_1[0]*quat_2[0] - quat_1[1]*quat_2[1] - quat_1[2]*quat_2[2] - quat_1[3]*quat_2[3]
    quat_out[1] = quat_1[0]*quat_2[1] + quat_1[1]*quat_2[0] + quat_1[2]*quat_2[3] - quat_1[3]*quat_2[2]
    quat_out[2] = quat_1[0]*quat_2[2] - quat_1[1]*quat_2[3] + quat_1[2]*quat_2[0] + quat_1[3]*quat_2[1]
    quat_out[3] = quat_1[0]*quat_2[3] + quat_1[1]*quat_2[2] - quat_1[2]*quat_2[1] + quat_1[3]*quat_2[0]
    return quat_out

def rotate(quat, point):
    """Rotate a point by the quaternion"""
    rotation_matrix = quaternion_to_matrix(quat)
    return _numpy.squeeze(_numpy.array(rotation_matrix*_numpy.transpose(_numpy.matrix(point))))

def rotate_array(quat, z_coordinates, y_coordinates, x_coordinates):
    """Rotate coordinate vectors x, y and z by the rotation defined by the quaternion."""
    rotation_matrix = quaternion_to_matrix(quat)
    out_matrix = rotation_matrix*_numpy.matrix([z_coordinates, y_coordinates, x_coordinates])
    out_array = _numpy.array(out_matrix)
    return out_array[0], out_array[1], out_array[2]

def rotate_array_bw(quat, x_coordinates, y_coordinates, z_coordinates):
    """Like rotate_array but with the coordintes index backwords. Do not use."""
    rotation_matrix = quaternion_to_matrix_bw(quat)
    out_matrix = rotation_matrix*_numpy.matrix([x_coordinates, y_coordinates, z_coordinates])
    out_array = _numpy.array(out_matrix)
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
    """The sampling parameter n corresponding to a certain number of rotations."""
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
    
