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
    

def quaternion_to_euler_angle(quat):
    """Dummy docstring"""
    euler = pylab.zeros(3)
    euler[0] = pylab.arctan2(2.0*(quat[0]*quat[1] + quat[2]*quat[3]),
                             1.0 - 2.0*(quat[1]**2 + quat[2]**2))
    euler[1] = pylab.arcsin(2.0*(quat[0]*quat[2] - quat[1]*quat[3]))
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
    return squeeze(array(m*transpose(matrix(point))))

def rotate_array(quat, x, y, z):
    m = quaternion_to_matrix(quat)
    out_matrix = m*pylab.matrix([x, y, z])
    out_array = pylab.array(out_matrix)
    return out_array[0], out_array[1], out_array[2]

