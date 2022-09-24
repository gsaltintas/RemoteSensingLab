Remote Sensing Lab - L1W1
Gül Sena Altintas

-----------

- Local mean is calculated by performing 2d convolution with a mean kernel.
- Local Standard Deviation is calculated according to the formulas in the hint (ie. from the local mean of the input and input squared, clipping from below by zero is performed to avoid negative values in the standard deviation.)
- Finally, Wallis filter is performed by using the two-fold formula (I*r1 + r0).

After implementing these operations, I realized that the input doesn't behave as expected to the default/recommended parameter settings. Then, I wrote a small random search on the b, c, target_mean, and target_std values in the intervals described in the Wallis algorithm pdf from the lecture material. I concluded that higher standard deviations perform better on the noisy image while recommended parameters perform better on the denoised image.

Final parameters used are as follows:
`b, c, target_mean, target_std = 0.982, 0.945, 123.909, 78.713`

The output images are listed as 
```
01-image_original_full_wallis_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.tif  # Original image, not cropped
01-image_original_wallis_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.tif       # Original image, cropped
histogram_originalcropped_wallis_err_12_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.png  # Original image cropped, histogram
02-image_denoised_full_wallis_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.tif  # Denoised image, not cropped
02-image_denoised_wallis_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.tif       # Denoised image, cropped
histogram_denoised_wallis_err_864_b-0.982_c-0.945_tm-123.909_ts-78.713_ws-21.png   # Denoised image cropped, histogram
03-image-full.tif 							      # Full image
histogram_full.png							      # Full image, histogram
```
