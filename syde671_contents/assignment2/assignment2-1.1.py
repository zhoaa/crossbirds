import functools
from pathlib import Path
import cv2
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import time


def fourLoop(image, kernel):
    kernel = np.flip(kernel, axis=(0, 1))
    output = np.zeros_like(image, dtype=np.float64)

    kernel_height_pad = math.floor(len(kernel)/2)
    kernel_width_pad = math.floor(len(kernel[0])/2)
    padded_image = np.pad(image, pad_width=((kernel_height_pad, kernel_height_pad), (kernel_width_pad, kernel_width_pad)))

    for i in range(kernel_height_pad, len(padded_image)-kernel_height_pad):
        for j in range(kernel_width_pad, len(padded_image[0])-kernel_width_pad):
            for k in range(len(kernel)):
                for l in range(len(kernel[0])):
                    output[i-kernel_height_pad, j-kernel_width_pad] += padded_image[i+k-kernel_height_pad][j+l-kernel_width_pad] * kernel[k][l]
    return output

def twoLoop(image, kernel):
    kernel = np.flip(kernel, axis=(0, 1))
    output = np.zeros_like(image, dtype=np.float64)

    kernel_height_pad = math.floor(len(kernel)/2)
    kernel_width_pad = math.floor(len(kernel[0])/2)
    padded_image = np.pad(image, pad_width=((kernel_height_pad, kernel_height_pad), (kernel_width_pad, kernel_width_pad)))

    for i in range(kernel_height_pad, len(padded_image)-kernel_height_pad):
        for j in range(kernel_width_pad, len(padded_image[0])-kernel_width_pad):
            output[i-kernel_height_pad, j-kernel_width_pad] = np.sum(np.multiply(padded_image[i-kernel_height_pad:i+kernel_height_pad+1, j-kernel_width_pad:j+kernel_width_pad+1], kernel))
    return output
            

input_image_path = Path(r"[INPUT HERE]")
input_image = cv2.imread(input_image_path , cv2.IMREAD_GRAYSCALE)


box_filter = np.full((9, 9), 1/81)
dX_filter = [[1, 0, -1]]
dY_filter = [[1], [0], [-1]]

start = time.perf_counter()
fourloop = fourLoop(input_image, box_filter)
end = time.perf_counter()
print("Four Loop time:", end-start)

start = time.perf_counter()
twoloop = twoLoop(input_image, box_filter)
end = time.perf_counter()
print("Two Loop time:", end-start)

start = time.perf_counter()
convolve = scipy.signal.convolve2d(input_image, box_filter, mode="same", boundary="fill", fillvalue=0)
end = time.perf_counter()
print("convolve2D time:", end-start)

print(np.allclose(fourloop, twoloop))
print("Max diff:", np.max(np.abs(fourloop.astype(float) - twoloop.astype(float))))

print(np.allclose(convolve, twoloop))
print("Max diff:", np.max(np.abs(convolve.astype(float) - twoloop.astype(float))))

dataset = [
    (input_image, "Initial Image"),
    (fourloop, "Four Loop"),
    (twoloop, "Two Loop"),
    (convolve, "convolve2d")
]

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, (img, title) in zip(axes, dataset):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=14, pad=10) 
    ax.axis("off")
    
plt.tight_layout()
plt.savefig("[OUTPUT HERE]", dpi=300, bbox_inches="tight")
plt.close()
