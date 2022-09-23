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








# No cheating :) - Daudt

def inv(A,x):
    out, _, _, _ = np.linalg.lstsq(A, x)
    return out

def rpc_inv(X0,x1,x2,coeff1,I1,O1,coeff2,I2,O2, n_it_max=250):

    X = np.array(X0.copy())
    Dl0 = 0

    counter = 0
    for _ in range(n_it_max):
        counter += 1
        X3 = inv(O1, X)

        V = X3[1]
        U = X3[0]
        W = X3[2]

        Xtil = np.array([
            1,
            V,
            U,
            W,
            V*U,
            V*W,
            U*W,
            V*V,
            U*U,
            W*W,
            U*V*W,
            V*V*V,
            V*U*U,
            V*W*W,
            U*V*V,
            U*U*U,
            U*W*W,
            V*V*W,
            W*U*U,
            W*W*W])

        difXU = [0, 0, 1, 0, V, 0, W, 0, 2*U, 0, V*W, 0, 2*U*V, 0, V*V, 3*U*U, W*W, 0, 2*U*W, 0]
        difXV = [0, 1, 0, 0, U, W, 0, 2*V, 0, 0, U*W, 3*V*V, U*U, W*W, 2*U*V, 0, 0, 2*V*W, 0, 0]
        difXW = [0, 0, 0, 1, 0, V, U, 0, 0, 2*W, U*V, 0, 0, 2*V*W, 0, 0, 2*U*W, V*V, U*U, 3*W*W]


        a1 = coeff1[:,0]
        b1 = coeff1[:,1]
        c1 = coeff1[:,2]
        d1 = coeff1[:,3]
        Pa1 = np.matmul(Xtil, a1)
        Pb1 = np.matmul(Xtil, b1)
        Pc1 = np.matmul(Xtil, c1)
        Pd1 = np.matmul(Xtil, d1)

        A = np.zeros((4,3))
        A[0,0] = (Pb1 ** -2) * (np.sum(difXU * a1) * Pb1 - np.sum(difXU * b1) * Pa1)
        A[0,1] = (Pb1 ** -2) * (np.sum(difXV * a1) * Pb1 - np.sum(difXV * b1) * Pa1)
        A[0,2] = (Pb1 ** -2) * (np.sum(difXW * a1) * Pb1 - np.sum(difXW * b1) * Pa1)
        A[1,0] = (Pd1 ** -2) * (np.sum(difXU * c1) * Pd1 - np.sum(difXU * d1) * Pc1)
        A[1,1] = (Pd1 ** -2) * (np.sum(difXV * c1) * Pd1 - np.sum(difXV * d1) * Pc1)
        A[1,2] = (Pd1 ** -2) * (np.sum(difXW * c1) * Pd1 - np.sum(difXW * d1) * Pc1)

        X3 = inv(O2, X)

        V = X3[1]
        U = X3[0]
        W = X3[2]

        Xtil = np.array([
            1,
            V,
            U,
            W,
            V*U,
            V*W,
            U*W,
            V*V,
            U*U,
            W*W,
            U*V*W,
            V*V*V,
            V*U*U,
            V*W*W,
            U*V*V,
            U*U*U,
            U*W*W,
            V*V*W,
            W*U*U,
            W*W*W])

        difXU = [0, 0, 1, 0, V, 0, W, 0, 2*U, 0, V*W, 0, 2*U*V, 0, V*V, 3*U*U, W*W, 0, 2*U*W, 0]
        difXV = [0, 1, 0, 0, U, W, 0, 2*V, 0, 0, U*W, 3*V*V, U*U, W*W, 2*U*V, 0, 0, 2*V*W, 0, 0]
        difXW = [0, 0, 0, 1, 0, V, U, 0, 0, 2*W, U*V, 0, 0, 2*V*W, 0, 0, 2*U*W, V*V, U*U, 3*W*W]

        a2 = coeff2[:,0]
        b2 = coeff2[:,1]
        c2 = coeff2[:,2]
        d2 = coeff2[:,3]
        Pa2 = np.matmul(Xtil, a2)
        Pb2 = np.matmul(Xtil, b2)
        Pc2 = np.matmul(Xtil, c2)
        Pd2 = np.matmul(Xtil, d2)

        A[2,0] = (Pb2 ** -2) * (np.sum(difXU * a2) * Pb2 - np.sum(difXU * b2) * Pa2)
        A[2,1] = (Pb2 ** -2) * (np.sum(difXV * a2) * Pb2 - np.sum(difXV * b2) * Pa2)
        A[2,2] = (Pb2 ** -2) * (np.sum(difXW * a2) * Pb2 - np.sum(difXW * b2) * Pa2)
        A[3,0] = (Pd2 ** -2) * (np.sum(difXU * c2) * Pd2 - np.sum(difXU * d2) * Pc2)
        A[3,1] = (Pd2 ** -2) * (np.sum(difXV * c2) * Pd2 - np.sum(difXV * d2) * Pc2)
        A[3,2] = (Pd2 ** -2) * (np.sum(difXW * c2) * Pd2 - np.sum(difXW * d2) * Pc2)

        Dx1 = inv(I1, x1) - inv(I1, rpc(X,coeff1,I1,O1))   
        Dx2 = inv(I2, x2) - inv(I2, rpc(X,coeff2,I2,O2)) 

        Dl = np.array([Dx1[1], Dx1[0], Dx2[1], Dx2[0]])

        grad =  inv(np.matmul(A.T,A), np.matmul(A.T, Dl))  #(A'*A)\A'*Dl

        DX = grad * [0.1, 0.1, 100.0]

        
        X[:3] += DX

        if np.abs(np.linalg.norm(Dl0)-np.linalg.norm(Dl)) < 1e-9:
            break

        Dl0 = Dl

    return np.expand_dims(X,1)


def deg2utm(X_in):
    X = X_in
    utmzone = []
    for i in range(X.shape[1]):
        la = X[0, i]
        lo = X[1, i]

        sa = 6378137.000000
        sb = 6356752.314245

        e2 = np.sqrt((sa ** 2) - (sb ** 2)) / sb
        e2sq = e2 ** 2
        c = (sa ** 2) / sb

        lat = la * (np.pi / 180)
        lon = lo * (np.pi / 180)

        Huso = np.fix((lo / 6) + 31)
        S = ((Huso * 6) - 183)
        deltaS = lon - (S * (np.pi / 180))
        
        if la < -72:
            letter = 'C'
        elif la < -64:
            letter = 'D'
        elif la < -56:
            letter = 'E'
        elif la < -48:
            letter = 'F'
        elif la < -40:
            letter = 'G'
        elif la < -32:
            letter = 'H'
        elif la < -24:
            letter = 'J'
        elif la < -16:
            letter = 'K'
        elif la < -8:
            letter = 'L'
        elif la < 0:
            letter = 'M'
        elif la < 8:
            letter = 'N'
        elif la < 16:
            letter = 'P'
        elif la < 24:
            letter = 'Q'
        elif la < 32:
            letter = 'R'
        elif la < 40:
            letter = 'S'
        elif la < 48:
            letter = 'T'
        elif la < 56:
            letter = 'U'
        elif la < 64:
            letter = 'V'
        elif la < 72:
            letter = 'W'
        else:
            letter = 'X'

        a = np.cos(lat) * np.sin(deltaS)
        epsilon = 0.5 * np.log((1 +  a) / (1 - a))
        nu = np.arctan(np.tan(lat) / np.cos(deltaS)) - lat
        v = (c / ((1 + (e2sq * (np.cos(lat)) ** 2))) ** 0.5) * 0.9996
        ta = (e2sq / 2) * epsilon ** 2 * (np.cos(lat)) ** 2
        a1 = np.sin(2 * lat)
        a2 = a1 * (np.cos(lat)) ** 2
        j2 = lat + (a1 / 2)
        j4 = ((3 * j2) + a2) / 4
        j6 = ((5 * j4) + (a2 * (np.cos(lat)) ** 2)) / 3
        alpha = (3 / 4) * e2sq
        beta = (5 / 3) * alpha ** 2
        gama = (35 / 27) * alpha ** 3
        Bm = 0.9996 * c * (lat - alpha * j2 + beta * j4 - gama * j6)
        xx = epsilon * v * (1 + (ta / 3)) + 500000
        yy = nu * v * (1 + ta) + Bm
        
        if yy < 0:
            yy += 9999999

        X[0, i] = xx
        X[1, i] = yy

        utmzone.append('{} {}'.format(str(int(Huso)).zfill(2), letter))

    return X, utmzone





