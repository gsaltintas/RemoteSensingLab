import numpy as np


def rpc(X, coeff, I, O):
    """Project 3D point X to image coordinates using rational polynomial coefficients (RPC)

    Args:
        X: 3D point coordinates in homogeneous coordinates (4x1 vector)
        coeff: RPC coefficients (20x4 matrix)
        I: RPC coefficients (3x3 matrix)
        O: RPC coefficients (4x4 matrix)

    Returns:
        x_img: image coordinates of input point in homogeneous coordinates (3x1 vector)
    """

    # Normalize X by multiplying the inverse of O with X
    # Look into numpy.matmul and numnpy.linalg.inv
    # print(np.linalg.inv(O).shape, X)
    Xnorm = np.matmul(np.linalg.inv(O), X)

    # Extract u, v, and w from Xnorm
    u, v, w = Xnorm[:3]

    # Build Xtilde vector from u, v, and w
    Xtilde = np.array([1, v, u, w, v*u,
                       v*w, u*w, v**2, u**2,
                       w**2, u*v*w, v**3,
                       v*u**2, v*w**2, v**2*u,
                       u**3, u*w**2, v**2*w,
                       u**2*w, w**3])

    # build a, b, c, and d vectors from coeff
    a = coeff[:, 0]
    b = coeff[:, 1]
    c = coeff[:, 2]
    d = coeff[:, 3]

    # Calculate l and s according to RPC formulas
    l = np.matmul(Xtilde, a) / np.matmul(Xtilde, b)
    s = np.matmul(Xtilde, c) / np.matmul(Xtilde, d)

    # Build a vector with s and l in homogeneous coordinates
    x_img_norm = np.array([s, l, 1])

    # Reverse normalization by multiplying I by x_norm
    x_img = np.matmul(I, x_img_norm)
    # print(Xtilde.shape, x_img_norm.shape, x_img, x_img.shape)

    return x_img
