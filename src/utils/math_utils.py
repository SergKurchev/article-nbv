import numpy as np

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (x, y, z, w).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm

def clip_vector(v, max_val):
    return np.clip(v, -max_val, max_val)
