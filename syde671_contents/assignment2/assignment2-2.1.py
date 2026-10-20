import functools
from pathlib import Path
import cv2
import numpy as np
import math
import scipy

import matplotlib.pyplot as plt

input_image_path = Path(r"[INPUT HERE]")
input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32)

sharpened_image = np.zeros_like(input_image)
resharpened_image = np.zeros_like(input_image)

gauss_1d = cv2.getGaussianKernel(15, 15.0)
gauss_2d = np.outer(gauss_1d, gauss_1d.T)
sharpen_kernel = -1 * gauss_2d
sharpen_kernel[7, 7] += 2

for i in range(3):
    resharpened_image[:, :, i] = scipy.signal.convolve2d(input_image[:, :, i], gauss_2d, mode="same", boundary="fill", fillvalue=0)

blurred_image = np.copy(resharpened_image)

for i in range(3):
    sharpened_image[:, :, i] = scipy.signal.convolve2d(input_image[:, :, i], sharpen_kernel, mode="same", boundary="fill", fillvalue=0)
    resharpened_image[:, :, i] = scipy.signal.convolve2d(resharpened_image[:, :, i], sharpen_kernel, mode="same", boundary="fill", fillvalue=0)

dataset = [
    (cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), "Initial Image"),
    (cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB), "Blurred Image"),
    (cv2.cvtColor(resharpened_image, cv2.COLOR_BGR2RGB), "Resharpened Image")
]

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for ax, (img, title) in zip(axes, dataset):
    img = np.clip(img / 255.0, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis("off")

plt.tight_layout()
plt.savefig("[OUTPUT PLOTS HERE]", dpi=300, bbox_inches="tight")
plt.close()