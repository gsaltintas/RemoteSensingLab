import numpy as np
from scipy.signal import correlate2d


def normalize(x_in):
    """Normalizes x_in to have mean 0 and L2 norm 1.

    Args:
        x_in: input vector

    Returns:
        x: normalized x_in
    """

    # Subtract mean
    # check if the axis argument makes sense
    # x = x_in - np.mean(x_in, axis=None)
    x = x_in - np.mean(x_in, axis=1)
    if np.linalg.norm(x) == 0:
        return x
    # Divide by norm, look into np.linalg.norm. Be careful not to divide by 0
    x /= np.linalg.norm(x)
    assert 1 + 1e-3 > np.linalg.norm(x) > 1 - 1e-3
    return x


def cross_correlation(template, search_area):
    """
    Find position in search_area with highest normalized cross correlation. Return coordinates and ncc.
    """

    # Normalize template
    template_norm = normalize(template)

    # Calculate loop indices given size of inputs
    ty, tx = template.shape
    sx = search_area.shape[1] - tx 
    sy = search_area.shape[0] - ty 

    # Initialize normalized cross correlation variable
    ncc = -np.ones((sx, sy))

    # Loop through all locations
    for i in range(sx):
        for j in range(sy):
            # Crop image from search_area with the same size of template at the appropriate position
            local_window = search_area[j:j+ty, i:i+tx]
            # print(ty, tx, sy, sx, local_window.shape, search_area.shape, template_norm.shape)
            # break
            # local_window = search_area[i:i+tx, j:j+ty]

            # Normalize
            local_window_norm = normalize(local_window)

            # Calculate normalized cross correlation (ncc),
            #  here i, j??, ncc[i, j] given
            ncc[i, j] = np.sum(template_norm * local_window_norm)
            # ncc[i, j] = np.sum(template_norm * local_window_norm)
    assert (ncc >= -1).all() and (ncc <= 1).all()
    # print(np.min(ncc))
    # Find position of maximum
    ncc_argmax = np.unravel_index(np.argmax(ncc, axis=None), ncc.shape)

    # Convert position into relative displacements
    dx = ncc_argmax[1]  # with respect to x0
    dy = ncc_argmax[0]  # with respect to y0

    return dx, dy, ncc[ncc_argmax]
