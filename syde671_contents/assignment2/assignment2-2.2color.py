import functools
from pathlib import Path
import cv2
import numpy as np
import math
import scipy
import matplotlib

matplotlib.use("macosx")
import matplotlib.pyplot as plt


def align_images(im1, im2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    points = []

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", param)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", im1.copy())
    display = im1.copy()
    cv2.setMouseCallback("Image", select_point, display)

    while len(points) < 2:
        cv2.waitKey(1)

    x1 = np.array([p[0] for p in points], dtype=float)
    y1 = np.array([p[1] for p in points], dtype=float)
    cx1 = np.mean(x1)
    cy1 = np.mean(y1)

    points.clear()
    display = im2.copy()
    cv2.imshow("Image", display)
    cv2.setMouseCallback("Image", select_point, display)

    while len(points) < 2:
        cv2.waitKey(1)

    x2 = np.array([p[0] for p in points], dtype=float)
    y2 = np.array([p[1] for p in points], dtype=float)
    cx2 = np.mean(x2)
    cy2 = np.mean(y2)

    cv2.destroyWindow("Image")

    tx = round((w1 / 2 - cx1) * 2)
    if tx > 0:
        im1 = np.pad(im1, ((0, 0), (tx, 0)) + ((0, 0),) * (im1.ndim - 2))
    else:
        im1 = np.pad(im1, ((0, 0), (0, -tx)) + ((0, 0),) * (im1.ndim - 2))

    ty = round((h1 / 2 - cy1) * 2)
    if ty > 0:
        im1 = np.pad(im1, ((ty, 0), (0, 0)) + ((0, 0),) * (im1.ndim - 2))
    else:
        im1 = np.pad(im1, ((0, -ty), (0, 0)) + ((0, 0),) * (im1.ndim - 2))

    tx = round((w2 / 2 - cx2) * 2)
    if tx > 0:
        im2 = np.pad(im2, ((0, 0), (tx, 0)) + ((0, 0),) * (im2.ndim - 2))
    else:
        im2 = np.pad(im2, ((0, 0), (0, -tx)) + ((0, 0),) * (im2.ndim - 2))

    ty = round((h2 / 2 - cy2) * 2)
    if ty > 0:
        im2 = np.pad(im2, ((ty, 0), (0, 0)) + ((0, 0),) * (im2.ndim - 2))
    else:
        im2 = np.pad(im2, ((0, -ty), (0, 0)) + ((0, 0),) * (im2.ndim - 2))

    len1 = math.sqrt((y1[1] - y1[0]) ** 2 + (x1[1] - x1[0]) ** 2)
    len2 = math.sqrt((y2[1] - y2[0]) ** 2 + (x2[1] - x2[0]) ** 2)
    dscale = len2 / len1

    if dscale < 1:
        im1 = cv2.resize(
            im1, None, fx=dscale, fy=dscale, interpolation=cv2.INTER_LINEAR
        )
    else:
        im2 = cv2.resize(
            im2, None, fx=1.0 / dscale, fy=1.0 / dscale, interpolation=cv2.INTER_LINEAR
        )

    theta1 = math.atan2(-(y1[1] - y1[0]), x1[1] - x1[0])
    theta2 = math.atan2(-(y2[1] - y2[0]), x2[1] - x2[0])
    dtheta = theta2 - theta1

    im1 = scipy.ndimage.rotate(im1, dtheta * 180 / math.pi, reshape=True, order=1)

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    minw = min(w1, w2)
    brd = (max(w1, w2) - minw) / 2
    if minw == w1:
        im2 = im2[:, math.ceil(brd) : w2 - math.floor(brd), ...]
        tx = tx - math.ceil(brd)
    else:
        im1 = im1[:, math.ceil(brd) : w1 - math.floor(brd), ...]
        tx = tx + math.ceil(brd)

    minh = min(h1, h2)
    brd = (max(h1, h2) - minh) / 2
    if minh == h1:
        im2 = im2[math.ceil(brd) : h2 - math.floor(brd), ...]
        ty = ty - math.ceil(brd)
    else:
        im1 = im1[math.ceil(brd) : h1 - math.floor(brd), ...]
        ty = ty + math.ceil(brd)

    return im1, im2


im1 = cv2.imread("[INPUT HERE]").astype(np.float32) / 255.0
im2 = cv2.imread("[INPUT HERE]").astype(np.float32) / 255.0

im2, im1 = align_images(im2, im1)

gauss_1d = cv2.getGaussianKernel(61, 10)
gauss_2d = np.outer(gauss_1d, gauss_1d.T)
high_filter = np.zeros_like(im2)

for i in range(3):
    blurred_channel = scipy.signal.convolve2d(im2[:, :, i], gauss_2d, mode="same")
    high_filter[:, :, i] = im2[:, :, i] - blurred_channel

gauss_1d = cv2.getGaussianKernel(27, 4.3)
gauss_2d = np.outer(gauss_1d, gauss_1d.T)
low_filter = np.zeros_like(im1)

for i in range(3):
    blurred_channel = scipy.signal.convolve2d(im1[:, :, i], gauss_2d, mode="same")
    low_filter[:, :, i] = blurred_channel

high_gray = cv2.cvtColor(high_filter, cv2.COLOR_BGR2GRAY)
low_gray = cv2.cvtColor(low_filter, cv2.COLOR_BGR2GRAY)

dataset = [
    (high_filter + low_filter, "Both in Color"),
    (high_gray + low_gray, "Both in Grayscale"),
    (high_gray[:, :, None] + low_filter, "Color Low Frequencies"),
    (high_filter + low_gray[:, :, None], "Color High Frequencies"),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

for ax, (img, title) in zip(axes, dataset):
    if img.ndim == 2:
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(img[:, :, ::-1], 0, 1))
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("[OUTPUT PLOT HERE]", dpi=300, bbox_inches="tight")
plt.close()
