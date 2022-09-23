# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 02/2021

import numpy as np


def rgb2ihs(img_rgb, eps=1e-12):
    """Transforms image from RGB color space to IHS color space.

    Args:
        img_rgb: Input RGB image with values between 0 and 1.
        eps: value used for numerical stability in division. Defaults to 1e-12.

    Returns:
        Image in IHS space.
    """

    # Split image channels for convenience
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    # Calculate intensity channel
    intensity = 1.0 * (r + g + b) / 3

    # Calculate hue channel
    # NOTE: numpy uses radians, not degreess
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b)*(g - b) + eps)
    mask = b <= g
    hue = np.arccos(numerator / denominator)  # reutrns in radians
    hue_degrees = np.degrees(hue)
    hue = hue_degrees
    np.putmask(hue, ~mask, 360.0 - hue_degrees)

    # Calculate saturation channel
    saturation = 1.0 - (3.0 / (r + g + b)) * np.min((r, g, b), axis=0)

    # Stack the three new channels into a single image
    img_ihs = np.stack((intensity, hue_degrees, saturation), 2)

    return img_ihs


def ihs2rgb(img_ihs, eps=1e-10):
    """Transforms image from IHS color space to RGB color space.

    Args:
        img_ihs: Input IHS image.

    Returns:
        Image in RGB space with values between 0 and 1.
    """

    # Split image channels for convenience
    intensity = img_ihs[:, :, 0]
    hue = img_ihs[:, :, 1]
    saturation = img_ihs[:, :, 2]

    # NOTE: you can use masks to deal with all pixels in a given sector simultaneously.
    # E.g.:
    # mask = 0.5 <= hue
    # r[mask] = intensity[mask] - saturation[mask] + 0.1
    #
    # A loop can also be used but it is suboptimal.

    # I suggest you first initialize the r, g, and b bands as if all pixels belonged to the RG sector. This
    # way you'll initialize all the necessary varialbles. You can then overwrite the pixels in  the GB and
    # BR sectors with the appropriate values.

    # RG sector
    rg_mask = (0 <= hue) & (hue < 120)
    gb_mask = (120 <= hue) & (hue < 240)
    br_mask = (240 <= hue) & (hue <= 360)
    hue = np.radians(hue)
    b = intensity*(1.0-saturation)
    r = intensity*(1.0 + saturation * np.cos(hue) /
                   (np.cos(1.0/3*np.pi - hue) + eps))
    g = 3.0*intensity - (r+b)

    # GB sector
    hue -= 2.0/3*np.pi
    np.putmask(r, gb_mask, intensity*(1-saturation))
    np.putmask(g, gb_mask, intensity*(1 - saturation *
               np.cos(hue) / (np.cos(1/3*np.pi - hue) + eps)))
    np.putmask(b, gb_mask, 3*intensity - (r+g))

    # BR sector
    hue -= 2/3*np.pi    # -120 deg (-240 from orig)
    np.putmask(g, br_mask, intensity*(1-saturation))
    np.putmask(b, br_mask, intensity*(1 + saturation *
               np.cos(hue) / (np.cos(1/3*np.pi - hue) + eps)))
    np.putmask(r, br_mask, 3*intensity - (g+b))

    # Stack r, g, and b to form RGB image and return
    img_rgb = np.stack((r, g, b), 2)
    np.clip(img_rgb, a_min=0, a_max=1, out=img_rgb)

    return img_rgb


def brovey_pansharpening(img_rgb, img_pan, scaling=3.0, eps=1e-12):
    """Brovey pansharpening.

    Args:
        img_rgb: Bilinear upsampled RGB image.
        img_pan: Panchromatic channel.
        scaling (optional): Scale output. Defaults to 3.0.
        eps (optional): Used for numerical stability in division. Defaults to 1e-12.

    Returns:
        Brovey pansharpened image.
    """

    # Calculate per-pixel RGB sum
    img_rgb_sum = np.sum(img_rgb, axis=-1) + eps

    # Apply Brovey pansharpening formula to each channel
    # Apply scaling for better visualization of the results
    # Ensure output is between 0 and 1
    # r_new = img_rgb[:, :, 0] / img_rgb_sum
    # since we divide by the sum, better scale again
    img_brovey = img_rgb.copy() * scaling
    img_brovey[:, :, 0] *= img_pan / img_rgb_sum
    img_brovey[:, :, 1] *= img_pan / img_rgb_sum
    img_brovey[:, :, 2] *= img_pan / img_rgb_sum

    np.clip(img_brovey, a_min=0, a_max=1, out=img_brovey)
    return img_brovey
