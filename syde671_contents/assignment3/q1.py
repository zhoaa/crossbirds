import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as bwdist


def warpGrid(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]).T
    warped_corners = H @ corners
    warped_corners = (warped_corners[:2] / warped_corners[2]).T
    x, y = warped_corners[:, 0], warped_corners[:, 1]
    min_x, max_x = np.floor(x).min().astype(int), np.ceil(x).max().astype(int)
    min_y, max_y = np.floor(y).min().astype(int), np.ceil(y).max().astype(int)
    out_h, out_w = max_y - min_y + 1, max_x - min_x + 1

    shifted_H = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]) @ H
    y, x = np.indices((out_h, out_w))
    source = np.linalg.inv(shifted_H) @ np.vstack(
        (x.ravel(), y.ravel(), np.ones(x.size))
    )
    source_x = (source[0] / source[2]).reshape(out_h, out_w)
    source_y = (source[1] / source[2]).reshape(out_h, out_w)
    return source_x, source_y


def warpImageNearestNeighbor(im, H):
    h, w = im.shape[:2]
    x, y = warpGrid(im, H)
    x, y = np.round(x).astype(int), np.round(y).astype(int)
    valid = (0 <= x) & (x < w) & (0 <= y) & (y < h)
    output = np.zeros(x.shape + im.shape[2:])
    output[valid] = im[np.clip(y[valid], 0, h - 1), np.clip(x[valid], 0, w - 1)]
    return output


def warpImageBilinear(im, H):
    h, w = im.shape[:2]
    x, y = warpGrid(im, H)
    dx, dy = x - np.floor(x), y - np.floor(y)
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    valid = (0 <= x) & (x + 1 < w) & (0 <= y) & (y + 1 < h)
    x, y = np.clip(x, 0, w - 2), np.clip(y, 0, h - 2)

    if im.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]

    values = (
        (1 - dx) * (1 - dy) * im[y, x]
        + dx * (1 - dy) * im[y, x + 1]
        + (1 - dx) * dy * im[y + 1, x]
        + dx * dy * im[y + 1, x + 1]
    )

    output = np.zeros(x.shape + im.shape[2:])
    output[valid] = values[valid]
    return output


def blendMosaic(left, warped_right, warped_mask, right_shape, H):
    h, w = right_shape[:2]
    corners = (
        H @ np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]).T
    )
    corners = corners[:2] / corners[2]
    min_x, min_y = np.floor(corners.min(axis=1)).astype(int)
    max_x, max_y = np.ceil(corners.max(axis=1)).astype(int)
    origin_x, origin_y = min(0, min_x), min(0, min_y)
    width = max(left.shape[1] - 1, max_x) - origin_x + 1
    height = max(left.shape[0] - 1, max_y) - origin_y + 1

    total = np.zeros((height, width, left.shape[2]))
    total_weight = np.zeros((height, width))

    left_x, left_y = -origin_x, -origin_y
    left_canvas = np.zeros_like(total)
    left_canvas[left_y : left_y + left.shape[0], left_x : left_x + left.shape[1]] = left
    left_mask = np.zeros((height, width), np.uint8)
    left_mask[left_y : left_y + left.shape[0], left_x : left_x + left.shape[1]] = 1
    left_weight = bwdist(left_mask)

    right_x, right_y = min_x - origin_x, min_y - origin_y
    right_canvas = np.zeros_like(total)
    right_canvas[
        right_y : right_y + warped_right.shape[0],
        right_x : right_x + warped_right.shape[1],
    ] = warped_right
    right_mask = np.zeros((height, width), np.uint8)
    right_mask[
        right_y : right_y + warped_mask.shape[0],
        right_x : right_x + warped_mask.shape[1],
    ] = (
        warped_mask > 0
    )
    right_weight = bwdist(right_mask)

    total = (
        left_canvas * left_weight[..., None] + right_canvas * right_weight[..., None]
    )
    total_weight = left_weight + right_weight

    return np.divide(
        total,
        total_weight[..., None],
        out=np.zeros_like(total),
        where=total_weight[..., None] != 0,
    ).astype(left.dtype)


def computeH(im1_pts, im2_pts):
    A = []
    b = []

    for (x1, y1), (x2, y2) in zip(im1_pts, im2_pts):
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        b.extend([x2, y2])

    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]])


# main because i call it in other q
def main():

    im1_pts = np.array(
        [
            [513.1, 393.0],
            [428.0, 651.2],
            [494.5, 682.2],
            [429.7, 463.1],
            [495.4, 453.4],
            [378.3, 175.7],
            [460.8, 71.9],
            [330.4, 238.7],
            [573.5, 578.5],
            [631.1, 364.7],
            [408.5, 619.3],
            [295.8, 413.5],
            [295.8, 464.0],
            [515.8, 937.7],
            [561.0, 558.0],
            [454.6, 565.1],
            [296.7, 283.0],
            [351.7, 210.3],
            [473.2, 196.1],
            [408.5, 252.0],
        ]
    )

    im2_pts = np.array(
        [
            [254.5, 419.7],
            [160.5, 687.6],
            [226.1, 716.0],
            [170.2, 488.0],
            [235.9, 480.9],
            [113.5, 181.9],
            [195.1, 94.1],
            [58.5, 237.8],
            [306.8, 606.8],
            [357.4, 402.8],
            [139.2, 654.7],
            [21.2, 426.8],
            [17.6, 482.6],
            [233.2, 976.8],
            [293.5, 585.5],
            [189.7, 594.4],
            [18.5, 279.5],
            [83.3, 210.3],
            [215.5, 218.3],
            [148.0, 265.3],
        ]
    )

    im1 = cv2.imread("ev3_left.jpg")
    im2 = cv2.imread("ev3_right.jpg")
    H = computeH(im2_pts, im1_pts)
    mask = np.ones(im2.shape[:2])

    warpedBi = warpImageBilinear(im2, H)
    warpedNN = warpImageNearestNeighbor(im2, H)
    warpedBiMask = warpImageBilinear(mask, H)
    warpedNNMask = warpImageNearestNeighbor(mask, H)

    cv2.imwrite("warpedBi.png", warpedBi)
    cv2.imwrite("warpedNN.png", warpedNN)
    cv2.imwrite("mosaicBi.png", blendMosaic(im1, warpedBi, warpedBiMask, im2.shape, H))
    cv2.imwrite("mosaicNN.png", blendMosaic(im1, warpedNN, warpedNNMask, im2.shape, H))

    original = cv2.imread("devin_resized.png")
    poster_corners = np.array([[319, 146], [668, 221], [675, 593], [320, 547]])
    rectangle = np.array([[0, 0], [399, 0], [399, 399], [0, 399]])
    rectification_H = computeH(poster_corners, rectangle)
    warped = warpImageBilinear(original, rectification_H)

    cv2.imwrite("devin_rectified.png", warped)
