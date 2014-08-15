import numpy

def random_quaternion():
    """Return a random rotation quaternion, as a length-4 array."""
    # The method of generating random rotations is taken from here:
    # http://planning.cs.uiuc.edu/node198.html
    random_base = numpy.random.random(3)
    quat = numpy.array([numpy.sqrt(1.-random_base[0])*numpy.sin(2.*numpy.pi*random_base[1]),
                        numpy.sqrt(1.-random_base[0])*numpy.cos(2.*numpy.pi*random_base[1]),
                        numpy.sqrt(random_base[0])*numpy.sin(2.*numpy.pi*random_base[2]),
                        numpy.sqrt(random_base[0])*numpy.cos(2.*numpy.pi*random_base[2])])
    return quat
    
def quaternion_from_dir_and_angle(angle, dir):
    q = numpy.zeros(4)
    #normalized_dir = numpy.array(dir)/norm(dir)
    q[0] = numpy.cos(angle/2.)
    q[1:] = numpy.array(dir)/numpy.linalg.norm(dir)*numpy.sqrt(1.-q[0]**2)
    return q

    

def quaternion_to_euler_angle(quat):
    """Generate euler angles from the quaternion. The last angle corresponds to in-plane rotation."""
    euler = numpy.zeros(3)
    euler[0] = numpy.arctan2(2.0*(quat[0]*quat[1] + quat[2]*quat[3]),
                             1.0 - 2.0*(quat[1]**2 + quat[2]**2))
    arcsin_argument = 2.0*(quat[0]*quat[2] - quat[1]*quat[3])
    if arcsin_argument > 1.0: arcsin_argument = 1.0
    if arcsin_argument < -1.0: arcsin_argument = -1.0
    euler[1] = numpy.arcsin(arcsin_argument)
    euler[2] = numpy.arctan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]),
                             1.0 - 2.0*(quat[2]**2 + quat[3]**2))
    return euler

def quaternion_to_matrix(quat):
    """Dummy docstring"""
    return numpy.matrix([[quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2,
                          2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3],
                          2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2],],
                         [2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3],
                          quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2,
                          2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1]],
                         [2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2],
                          2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1],
                          quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2]])

def quaternion_inverse(quat):
    q = numpy.zeros(4)
    q[0] = quat[0]
    q[1:] = -quat[1:]
    return q

def quaternion_multiply(quat_1, quat_2):
    q = numpy.zeros(4)
    # q[0] = quat_1[0]*quat_2[0] - quat_1[1]*quat_2[1] - quat_1[2]*quat_2[2] - quat_1[3]*quat_2[3]
    # q[1] = quat_1[1]*quat_2[0] + quat_1[0]*quat_2[1] + quat_1[3]*quat_2[2] - quat_1[2]*quat_2[3]
    # q[2] = quat_1[2]*quat_2[0] + quat_1[0]*quat_2[2] - quat_1[3]*quat_2[1] + quat_1[1]*quat_2[3]
    # q[3] = quat_1[3]*quat_2[0] + quat_1[2]*quat_2[1] - quat_1[1]*quat_2[2] + quat_1[0]*quat_2[3]
    q[0] = quat_1[0]*quat_2[0] - quat_1[1]*quat_2[1] - quat_1[2]*quat_2[2] - quat_1[3]*quat_2[3]
    q[1] = quat_1[0]*quat_2[1] + quat_1[1]*quat_2[0] + quat_1[2]*quat_2[3] - quat_1[3]*quat_2[2]
    q[2] = quat_1[0]*quat_2[2] - quat_1[1]*quat_2[3] + quat_1[2]*quat_2[0] + quat_1[3]*quat_2[1]
    q[3] = quat_1[0]*quat_2[3] + quat_1[1]*quat_2[2] - quat_1[2]*quat_2[1] + quat_1[3]*quat_2[0]

    return q

def rotate(quat, point):
    m = quaternion_to_matrix(quat)
    return numpy.squeeze(numpy.array(m*numpy.transpose(numpy.matrix(point))))

def rotate_array(quat, z, y, x):
    m = quaternion_to_matrix(quat)
    out_matrix = m*numpy.matrix([z, y, x])
    out_array = numpy.array(out_matrix)
    return out_array[0], out_array[1], out_array[2]

def read_quaternion_list(filename):
    import h5py
    with h5py.File(filename) as file_handle:
        number_of_rotations = file_handle['number_of_rotations'][...]
        quaternions = numpy.zeros((number_of_rotations, 4))
        weights = numpy.zeros(number_of_rotations)
        for i in range(number_of_rotations):
            quaternion_and_weight = file_handle[str(i)][...]
            quaternions[i] = quaternion_and_weight[:4]
            weights[i] = quaternion_and_weight[4]
    return quaternions, weights

def n_to_rots(n):
    return 10*(n+5*n**3)

def n_to_angle(n):
    tau = (1.0 + numpy.sqrt(5.0))/2.0
    return 4./n/tau**3

def rots_to_n(rots):
    n = 1
    while True:
        this_rots = n_to_rots(n)
        if this_rots == rots:
            return n
        if this_rots > rots:
            raise ValueError("%d rotations does not correspond to any n" % rots)
        n += 1

def quaternion_to_angle(quat):
    """The angle by which this transformation rotates"""
    return 2.*numpy.arccos(quat[0])

def quaternion_to_axis(quat):
    """The axis around which this rotation rotates"""
    return quat[1:]/norm(quat[1:])
