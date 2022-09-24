# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 02/2021


################################################################################
############################## Lab 1 - Week 1 ##################################
################################################################################
# 
# Tasks:
# 
# 1. In util.py:
#     - Fill in code for local_mean()
#     - Fill in code for local_std()
#     - Fill in code for wallis()
# 
# 2. Find appropriate filtering parameters using a cropped version of the image for speed. What effect does each parameter have?
# 
# 3. Compare results of Wallis filtering on image before and after denoising
# 
################################################################################

import numpy as np
from scipy import signal
import skimage


def local_mean(I, window_size : int = 21):
    """
    Windowed mean calculation.

    Args:
        I: input 2d image (grayscale).
        window_size (int): kernel size, must be an odd number.

    Returns:
        I_mean: local mean around each pixel
    """
    
    # Ensure window size is odd and positive
    assert(window_size % 2 == 1)
    assert(window_size > 0)
    
        
    # TODO: define a square mean kernel, check numpy.ones
    # Tip: the sum of the kernel elements must be 1
    # each entry in the input contributes equally
    kernel = np.ones((window_size, window_size)) / window_size ** 2
    
    # TODO: Calculate local mean using a convolution, check scipy.signal.convolve2d
    # Tip: optional arguments 'mode' and 'boundary' are important here
    # Tip: output should have the same shape as input
    # not sure how to use boundary
    return signal.convolve2d(I, kernel, mode='same', boundary='symm')
    return signal.convolve2d(I, kernel, mode='same', boundary='fill')
    return signal.convolve2d(I, kernel, mode='same', boundary='wrap')

    region = skimage.morphology.rectangle(window_size, window_size)
    return skimage.filters.rank.mean(I, footprint=region)
    
    
    
    # Alternatively you can use skimage.morphology.square and skimage.filters.rank.mean
#     return skimage.rank.mean(I, footprint=)
    
    # If you're not comforable with convolutions this can be done with nested loops, but it will probably be slower
    
    
    print('IF YOU\'RE READING THIS, GO INTO UTILS.PY AND FILL IN THIS FUNCTION')
    
    return np.ones(I.shape)


def local_std(I, window_size : int = 21, E_x = None):
    """
    Windowed standard deviation calculation.
    To be used as Matlab's stdfilt.

    Args:
        I: input 2d image (grayscale).
        window_size (int): kernel size, must be an odd number.
        E_x: precomputed local mean if available.

    Returns:
        I_std: local standard deviation around each pixel
    """
    
    # Ensure window size is odd and positive
    assert(window_size % 2 == 1)
    assert(window_size > 0)

        
    # TODO: calculate local mean of I if not provided (use local_mean)
    if E_x is None:
        E_x = local_mean(I, window_size)
        
    # TODO: calculate local mean of I**2 (use local_mean)
    I_std = local_mean(I**2, window_size)
    
    # TODO: use the identity Var(x) = E[X**2] - E[X]**2 to calculate I_std
    I_std -= E_x**2
    
    # Tip: Clipping before sqrt may be necessary to avoid negative values due to numerical errors
    np.clip(I_std, a_min=0, a_max=None, out=I_std)
    return I_std ** 0.5
    
    print('\n\nIF YOU\'RE READING THIS, GO INTO UTILS.PY AND FILL IN THIS FUNCTION\n\n')
    
    return np.ones(I.shape)


def wallis(I, I_mean=None, I_std=None, b : float = 1.0, c : float = 0.94, target_mean : float = 127, target_std : float = 40):
    """
    Apply Wallis filter to I.

    Args:
        I: Image do be filtered
        I_mean: Local mean
        I_std: Local standard deviation
        b (float, optional): Brightness enforcing constant. Defaults to 1.0.
        c (float, optional): Contrast enforcing constant. Defaults to 0.94.
        target_mean (int, optional): Target mean. Defaults to 127.
        target_std (int, optional): Target standard deviation. Defaults to 40.

    Returns:
        I_wallis: Output of Wallis filter.
    """
    
    # Sanity checks
    assert(len(I.shape) == 2)
    assert(len(I_mean.shape) == 2)
    assert(len(I_std.shape) == 2)
    assert(I.shape[0] == I_mean.shape[0])
    assert(I.shape[0] == I_std.shape[0])
    assert(I.shape[1] == I_mean.shape[1])
    assert(I.shape[1] == I_std.shape[1])

    
    # TODO: apply Wallis filter to I
    # Tip: make sure output is between 0 and 255, check documentation for numpy.clip
    # Tip: if you're not comfortable with numpy broadcasting it may be easier to use nested for loops

    r1 = c*target_std / (c*I_std + (1-c)*target_std)
    r0 = b*target_mean + (1 - b - r1) * I_mean
    output = I*r1 + r0
    np.clip(output, a_min=0, a_max=255, out=output)
    return output
    
    print('\n\nIF YOU\'RE READING THIS, GO INTO UTILS.PY AND FILL IN THIS FUNCTION\n\n')

    return np.ones(I.shape)




