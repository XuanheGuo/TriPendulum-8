import numpy as np

def wrap_angle(angle):
    """
    Wraps an angle (in radians) or numpy array of angles to the range [-pi, pi].
    """
    angle = np.asarray(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi

def relative_to_absolute(q):
    """
    Converts relative joint angles q to absolute angles theta_abs.
    q pos in MuJoCo:
      q1 = angle of pole1 relative to the vertical line
      q2 = angle of pole2 relative to pole1
      q3 = angle of pole3 relative to pole2
      
    Absolute angles relative to vertical:
      theta1_abs = q1
      theta2_abs = q1 + q2
      theta3_abs = q1 + q2 + q3
    """
    q_arr = np.asarray(q)
    return np.cumsum(q_arr, axis=-1)

def angle_error(theta, theta_goal):
    """
    Computes the wrapped error between theta and theta_goal.
    Resulting errors will be in [-pi, pi].
    """
    theta = np.asarray(theta)
    theta_goal = np.asarray(theta_goal)
    return wrap_angle(theta - theta_goal)
