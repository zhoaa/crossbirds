import functools
from pathlib import Path
import cv2
import numpy as np
import math
import scipy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

input_image_path = Path(r"[INPUT HERE]")
input_image = cv2.imread(input_image_path , cv2.IMREAD_GRAYSCALE)
dX_filter = [[1, 0, -1]]
dY_filter = [[1], [0], [-1]]

I_x = scipy.signal.convolve2d(input_image, dX_filter, mode="same")
I_y = scipy.signal.convolve2d(input_image, dY_filter, mode="same")
gradient_mag = np.sqrt(I_x**2 + I_y**2)
thresholded = np.where(gradient_mag > 40, 255, 0)
cv2.imwrite("[OUTPUT THRESHOLDED HERE]", thresholded)

gauss_1d = cv2.getGaussianKernel(9, 1.5) 
gauss_2d = np.outer(gauss_1d, gauss_1d.T)
blurred = scipy.signal.convolve2d(input_image, gauss_2d, mode="same")
I_x = scipy.signal.convolve2d(blurred, dX_filter, mode="same")
I_y = scipy.signal.convolve2d(blurred, dY_filter, mode="same")
gradient_mag = np.sqrt(I_x**2 + I_y**2)
thresholded_blurred = np.where(gradient_mag > 40, 255, 0)
cv2.imwrite("[OUTPUT BLURRED HERE]", thresholded_blurred)

convolved_gaussian_X = scipy.signal.convolve2d(gauss_2d, dX_filter, mode="full")
convolved_gaussian_Y = scipy.signal.convolve2d(gauss_2d, dY_filter, mode="full")
I_x = scipy.signal.convolve2d(input_image, convolved_gaussian_X, mode="same")
I_y = scipy.signal.convolve2d(input_image, convolved_gaussian_Y, mode="same")
gradient_mag = np.sqrt(I_x**2 + I_y**2)
thresholded_single_conv = np.where(gradient_mag > 40, 255, 0)
cv2.imwrite("[OUTPUT SINGLE CONVOLVE BLURRED HERE]", thresholded_single_conv)

orientation = np.arctan2(I_y, I_x)
hsv = np.zeros((*I_x.shape, 3), dtype=np.float32)
hsv[:, :, 0] = (orientation + np.pi) / (2 * np.pi)
hsv[:, :, 1] = 1.0  
hsv[:, :, 2] = 1.0 


rgb = mcolors.hsv_to_rgb(hsv)
plt.imshow(rgb)
plt.savefig("[OUTPUT HSV HERE]", bbox_inches="tight")


dataset = [
    (input_image, "Initial Image"),
    (thresholded, "Convolved w/ Dx, Dy"),
    (thresholded_blurred, "Blurred"),
    (thresholded_single_conv, "Single Convolution"),
    
]

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, (img, title) in zip(axes, dataset):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=14, pad=10) 
    ax.axis("off")
    
plt.tight_layout()
output_filename = "[OUTPUT PLOTS HERE]"
plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.close()
