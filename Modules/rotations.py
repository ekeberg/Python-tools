import pylab

def random_quaternion():
    """Return a random rotation quaternion, as a length-4 array."""
    #create a random quaternion
    quat = -1. + 2.*pylab.rand(4)
    #normalize it
    quat /= pylab.norm(quat)
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
