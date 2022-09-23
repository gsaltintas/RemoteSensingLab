# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 02/2021

# IMPORTANT: to use this function you may need to use matplotlib.use('TkAgg') or something similar

import matplotlib.pyplot as plt
import numpy as np
from time import sleep

def click_point(I, p_approx, window_radius=25, title_addendum='', from_notebook=True):
    """Return image coordinates of clicked point.

    Args:
        I: Full image.
        p_approx: approximate position of point of interest.
        window_radius (int, optional): window radius to crop patch. Defaults to 25.
        title_addendum (str, optional): string to be added to the title for clarity.

    Returns:
        p: image coordinates of clicked point.
    """
    
    # Calculate patch boundaries
    x1 = int(max(p_approx[1] - window_radius, 0))
    x2 = int(min(p_approx[1] + window_radius, I.shape[0]))
    y1 = int(max(p_approx[0] - window_radius, 0))
    y2 = int(min(p_approx[0] + window_radius, I.shape[1]))
    
    # Crop patch centered in expected point position
    crop = I[x1:x2+1, y1:y2+1]
        
    # Let user click on point
    plt.figure()
    plt.title('Click on GCP' + title_addendum)
    plt.imshow(crop, cmap='gray')
    if from_notebook:
        plt.show() # this blocks interactivity when running python script directly
    offset = np.asarray(plt.ginput(1, timeout=0)) - window_radius
    sleep(0.3)
    plt.close()
    
    # Calculate image coordinates of clicked point
    p = p_approx
    p[0] -= offset[0,1]
    p[1] -= offset[0,0]
    
    return p
    