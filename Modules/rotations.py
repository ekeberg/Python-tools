import pylab

def random_quaternion():
    """Return a random rotation quaternion, as a length-4 array."""
    # The method of generating random rotations is taken from here:
    # http://planning.cs.uiuc.edu/node198.html
    random_base = pylab.rand(3)
    quat = pylab.array([pylab.sqrt(1.-random_base[0])*pylab.sin(2.*pylab.pi*random_base[1]),
                        pylab.sqrt(1.-random_base[0])*pylab.cos(2.*pylab.pi*random_base[1]),
                        pylab.sqrt(random_base[0])*pylab.sin(2.*pylab.pi*random_base[2]),
                        pylab.sqrt(random_base[0])*pylab.cos(2.*pylab.pi*random_base[2])])
    return quat
    
def quaternion_from_dir_and_angle(angle, dir):
    q = pylab.zeros(4)
    #normalized_dir = pylab.array(dir)/norm(dir)
    q[0] = pylab.cos(angle/2.)
    q[1:] = pylab.array(dir)/pylab.norm(dir)*pylab.sqrt(1.-q[0]**2)
    return q

    

def quaternion_to_euler_angle(quat):
    """Generate euler angles from the quaternion. The last angle corresponds to in-plane rotation."""
    euler = pylab.zeros(3)
    euler[0] = pylab.arctan2(2.0*(quat[0]*quat[1] + quat[2]*quat[3]),
                             1.0 - 2.0*(quat[1]**2 + quat[2]**2))
    arcsin_argument = 2.0*(quat[0]*quat[2] - quat[1]*quat[3])
    if arcsin_argument > 1.0: arcsin_argument = 1.0
    if arcsin_argument < -1.0: arcsin_argument = -1.0
    euler[1] = pylab.arcsin(arcsin_argument)
    euler[2] = pylab.arctan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]),
                             1.0 - 2.0*(quat[2]**2 + quat[3]**2))
    return euler

def quaternion_to_matrix(quat):
    """Dummy docstring"""
    return pylab.matrix([[quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2,
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
    return pylab.matrix([[quat[0]**2-quat[1]**2-quat[2]**2+quat[3]**2,
                          2.0*quat[2]*quat[3]+2.0*quat[0]*quat[1],
                          2.0*quat[1]*quat[3]-2.0*quat[0]*quat[2]],
                         [2.0*quat[2]*quat[3]-2.0*quat[0]*quat[1],
                          quat[0]**2-quat[1]**2+quat[2]**2-quat[3]**2,
                          2.0*quat[1]*quat[2]+2.0*quat[0]*quat[3]],
                         [2.0*quat[1]*quat[3]+2.0*quat[0]*quat[2],
                          2.0*quat[1]*quat[2]-2.0*quat[0]*quat[3],
                          quat[0]**2+quat[1]**2-quat[2]**2-quat[3]**2]])

def quaternion_inverse(quat):
    q = pylab.zeros(4)
    q[0] = quat[0]
    q[1:] = -quat[1:]
    return q

def quaternion_multiply(quat_1, quat_2):
    q = pylab.zeros(4)
    q[0] = quat_1[0]*quat_2[0] - quat_1[1]*quat_2[1] - quat_1[2]*quat_2[2] - quat_1[3]*quat_2[3]
    q[1] = quat_1[1]*quat_2[0] + quat_1[0]*quat_2[1] + quat_1[3]*quat_2[2] - quat_1[2]*quat_2[3]
    q[2] = quat_1[2]*quat_2[0] + quat_1[0]*quat_2[2] - quat_1[3]*quat_2[1] + quat_1[1]*quat_2[3]
    q[3] = quat_1[3]*quat_2[0] + quat_1[2]*quat_2[1] - quat_1[1]*quat_2[2] + quat_1[0]*quat_2[3]
    return q

def rotate(quat, point):
    m = quaternion_to_matrix(quat)
    return pylab.squeeze(pylab.array(m*pylab.transpose(pylab.matrix(point))))

def rotate_array(quat, x, y, z):
    m = quaternion_to_matrix(quat)
    out_matrix = m*pylab.matrix([x, y, z])
    out_array = pylab.array(out_matrix)
    return out_array[0], out_array[1], out_array[2]

def rotate_array_bw(quat, x, y, z):
    m = quaternion_to_matrix_bw(quat)
    out_matrix = m*pylab.matrix([x, y, z])
    out_array = pylab.array(out_matrix)
    return out_array[0], out_array[1], out_array[2]

def read_quaternion_list(filename):
    import h5py
    with h5py.File(filename) as file_handle:
        number_of_rotations = file_handle['number_of_rotations'][...]
        quaternions = pylab.zeros((number_of_rotations, 4))
        weights = pylab.zeros(number_of_rotations)
        for i in range(number_of_rotations):
            quaternion_and_weight = file_handle[str(i)][...]
            quaternions[i] = quaternion_and_weight[:4]
            weights[i] = quaternion_and_weight[4]
    return quaternions, weights

def n_to_rots(n):
    return 10*(n+5*n**3)

def n_to_angle(n):
    tau = (1.0 + pylab.sqrt(5.0))/2.0
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
