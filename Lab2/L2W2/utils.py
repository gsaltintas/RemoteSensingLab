# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 04/2021

import math
import numpy as np
from tqdm import tqdm

def pdf(x, mu, sigma):
    """Evaluate multivariate probability function for x.

    Args:
        x: Input image. For an M by N image and C channels, x must be of shape [M, N, C]
        mu: Mean vector for the considered class of shape [C] or [C, 1]
        sigma: Covariance matrix for the considered class of size [C, C]

    Returns:
        Multivariate probability density function evaluated for each pixel in x.
    """

    # Precompute everything that doesn't change per point
    # This is important to reduce computing times

    C = x.shape[2]
    c = 1. / np.sqrt(((2*math.pi) ** C) * np.linalg.det(sigma)) # multiplicative factor (one over sqrt of ...)
    sigma_inv = np.linalg.inv(sigma) # inverse of matrix sigma

    x_norm = x - mu # remove mean from all points using a broadcasting operation to reduce the load in for loop
    probs = np.zeros((x.shape[0], x.shape[1])) # initialize variable where probabilities will be stored

    # loop through all pixels
    for i in tqdm(range(x.shape[0])):
        for j in range(x.shape[1]):
            p = x_norm[i, j, :].T # reshape x_norm[i, j, :] into a Cx1 vector
            # Note: matrix multiplication is NOT done using *
            exponent = -1. / 2 * np.matmul(np.matmul(p.T, sigma_inv), p) # calculate the exponent using the provided formula
            probs[i, j] = c * np.exp(exponent) # calculate the probability using the provided formula
    
    return probs

    

