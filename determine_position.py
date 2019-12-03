import numpy as np

def calculate_T_WC(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """
    input: coefficients of three plane functions in world coordinates.
    output: T_WC, T_WC * p_C = p_W
    """
    # first find the rotation matrix associated with these values
    def find_vector(a1, b1, c1, a2, b2, c2):
        """
        find the unit vector parallel to intersected line of two planes
        """
        delta_x = (c2 - c1) / (c2 * a1 - c1 * a2)
        delta_y = (c1 - c2) / (c2 * b1 - c1 * b2)
        delta_z = (a1 - a2) / (c2 * a1 - c1 * a2) - (b1 - b2) / (c2 * b1 - c1 * b2)
        if delta_x < 0: # reverse all signs. We are assuming the sign is same as world frame
            delta_x = - delta_x
            delta_y = - delta_y
            delta_z = - delta_z
        # now normalize x, y, z
        norm = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
        return delta_x / norm, delta_y / norm, delta_z / norm
    
    x2, y2, z2 = find_vector(a1, b1, c1, a2, b2, c2)
    x1, y1, z1 = find_vector(a1, b1, c1, a3, b3, c3)
    x3, y3, z3 = find_vector(a2, b2, c2, a3, b3, c3)

    R = np.array([
        [x1, x2, x3],
        [y1, y2, y3],
        [z1, z2, z3],
    ]) # the R is not perfect, need to do svd
    u, s, vh = np.linalg.svd(R) # svd to ensure orthogonality
    R = np.matmul(u, vh)

    # next, find the intersection of three planes
    A = np.array([
        [a1, b1, c1],
        [a2, b2, c2],
        [a3, b3, c3],
    ])
    b = np.array([[1, 1, 1]]).T
    p = np.linalg.solve(A, b).squeeze()

    T = np.array([
        [x1, x2, x3, p[0]],
        [y1, y2, y3, p[1]],
        [z1, z2, z3, p[2]],
        [0,  0,  0,  1]
    ])

    return T