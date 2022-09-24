import numpy as np

def pointwise_RMSE(p1, p2):
    return np.sqrt(np.mean((p1 - p2)**2, axis=0))

def get_affine_transform(set_1, set_2):
    """Calculate affine transform that best fits set_1 ~= Aff(set_2)

    Args:
        set_1: first set of points (3xN)
        set_2: second set of points (3xN)

    Returns:
        Aff: Affine transform matrix
        res: point wise residuals
        RMSE: Root mean square residuals
    """
    
    num_points = set_1.shape[1]
    B = np.zeros((2*num_points, 6))
    B[:num_points,:2] = set_2[:2,:].T
    B[:num_points,4] = np.ones(num_points)
    B[num_points:2*num_points,2:4] = set_2[:2,:].T
    B[num_points:2*num_points,5] = np.ones(num_points)
    
    y = np.ravel(set_1[:2,:])
    
    x = np.matmul(np.linalg.pinv(B), y)
    
    
    Aff = np.zeros((3,3))
    Aff[0,0] = x[0]
    Aff[0,1] = x[1]
    Aff[0,2] = x[4]
    Aff[1,0] = x[2]
    Aff[1,1] = x[3]
    Aff[1,2] = x[5]
    Aff[2,2] = 1
    
    
    res = set_1 - np.matmul(Aff, set_2)
    res = res[:2,:]
    RMSE = np.sqrt(np.mean(res ** 2))

    return Aff, res, RMSE

def get_affine_residuals(set_1, set_2, affine):
    residuals = set_1 - np.matmul(affine, set_2)
    return residuals[:2,:]