import math
from pathlib import Path
import cv2
import numpy as np


def ncc(A, B):
    # A, B are grayscale images (matrices)
    a = A.astype(np.float32).ravel()
    b = B.astype(np.float32).ravel()

    # normalize for overall brightness by subtracting the mean
    a -= a.mean()
    b -= b.mean()

    # use dot product to find how similar vectors are
    ncc = np.dot(a, b)

    # normalize for contrast by dividing by the magnitude
    ncc = ncc / np.sqrt(np.dot(a, a) * np.dot(b, b))

    return ncc


def align(anchor, cand):
    # find the number of levels, flooring at about 200 pixels
    levels = max(1, int(math.log2(min(anchor.shape[:2]) / 200)) + 1)

    # store the image pyramids
    anchor_pyr = [anchor.astype(np.float32)]
    cand_pyr = [cand.astype(np.float32)]

    for _ in range(1, levels):
        # blur before downsampling to reduce aliasing
        anchor_blur = cv2.GaussianBlur(anchor_pyr[-1], (5, 5), 0)
        cand_blur = cv2.GaussianBlur(cand_pyr[-1], (5, 5), 0)

        # downsample by a factor of 2
        anchor_pyr.append(anchor_blur[::2, ::2])
        cand_pyr.append(cand_blur[::2, ::2])

    dx = 0
    dy = 0
    best_score = -1

    for level in range(levels - 1, -1, -1):

        # scale the previous shift to the current resolution
        if level < levels - 1:
            dx *= 2
            dy *= 2

        anchor_cur = anchor_pyr[level]
        cand_cur = cand_pyr[level]

        # search farther at the coarsest level
        radius = 25 if level == levels - 1 else 5

        best_score = -1
        best_shift = (dx, dy)

        # search a small neighborhood around the current alignment
        for ddy in range(-radius, radius + 1):
            for ddx in range(-radius, radius + 1):

                cand_dx = dx + ddx
                cand_dy = dy + ddy

                y0 = max(0, cand_dy)
                y1 = min(anchor_cur.shape[0], anchor_cur.shape[0] + cand_dy)

                x0 = max(0, cand_dx)
                x1 = min(anchor_cur.shape[1], anchor_cur.shape[1] + cand_dx)

                # compare only the overlapping regions
                a = anchor_cur[y0:y1, x0:x1]
                b = cand_cur[y0 - cand_dy : y1 - cand_dy, x0 - cand_dx : x1 - cand_dx]

                score = ncc(a, b)

                if score > best_score:
                    best_score = score
                    best_shift = (cand_dx, cand_dy)

        dx, dy = best_shift

    return best_score, (dx, dy)


def split_images(image):
    h = image.shape[0]

    # compute the mean intensity of every row
    profile = image.astype(np.float32).mean(axis=1)
    d = max(2, h // 150)

    # find dark rows that could separate the three exposures
    trough = np.zeros(h)
    trough[d : h - d] = (profile[: h - 2 * d] + profile[2 * d :]) / 2 - profile[
        d : h - d
    ]

    best = (-np.inf, 0, h // 3)

    # search for the two gaps between the exposures
    for pitch in range(int(h / 3.15), h // 3 + 1):
        n = h - 3 * pitch + 1

        score = np.minimum(trough[pitch : pitch + n], trough[2 * pitch : 2 * pitch + n])

        start = np.argmax(score)

        if score[start] > best[0]:
            best = (score[start], start, pitch)

    _, start, pitch = best

    return [image[start + k * pitch : start + (k + 1) * pitch] for k in range(3)]


def detect_common_crop(images):

    # handpicked values for the inner and outer black and white respectively
    inner_border = 64
    outer_border = 245

    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)

    # stack the images so their borders can be compared
    stack = np.stack([p[:h, :w] for p in images]).astype(np.float32)

    # mark pixels that look like black or white border material
    borderline = (stack < inner_border) | (stack > outer_border)

    rows = borderline.mean(axis=2).max(axis=0) > 0.5
    cols = borderline.mean(axis=1).max(axis=0) > 0.5

    # extend border detections to cover slightly thicker edges
    k = max(3, int(0.02 * h))
    rows = np.logical_or.reduce(
        [np.r_[rows, np.zeros(k, bool)][i : i + h] for i in range(k)]
    )

    k = max(3, int(0.02 * w))
    cols = np.logical_or.reduce(
        [np.r_[cols, np.zeros(k, bool)][i : i + w] for i in range(k)]
    )

    # remove border rows and columns near the outside
    top = 0
    while top < 0.1 * h and rows[top]:
        top += 1

    left = 0
    while left < 0.1 * w and cols[left]:
        left += 1

    bottom = h
    while bottom > 0.9 * h and rows[bottom - 1]:
        bottom -= 1

    right = w
    while right > 0.9 * w and cols[right - 1]:
        right -= 1

    return left, top, right, bottom


input_folder = Path(r"[INPUT FOLDER]")
output_folder = Path(r"[OUTPUT FOLDER]")

for path in sorted(input_folder.iterdir()):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    B, G, R = split_images(image)

    # find the common valid region of all three images
    x0, y0, x1, y1 = detect_common_crop([B, G, R])

    B_cropped = B[y0:y1, x0:x1]
    G_cropped = G[y0:y1, x0:x1]
    R_cropped = R[y0:y1, x0:x1]

    # find the best alignment using the common crop
    g_ncc, (gdx, gdy) = align(B_cropped, G_cropped)
    r_ncc, (rdx, rdy) = align(B_cropped, R_cropped)

    print("Image:", path.name)
    print("G offset:", str((gdx, gdy)) + ", R offset:", str((rdx, rdy)))

    # align the cropped images
    G_cropped = np.roll(G_cropped, (gdy, gdx), axis=(0, 1))
    R_cropped = np.roll(R_cropped, (rdy, rdx), axis=(0, 1))

    result = cv2.merge([B_cropped, G_cropped, R_cropped])

    # remove pixels that do not overlap after alignment
    mx = max(abs(gdx), abs(rdx))
    my = max(abs(gdy), abs(rdy))

    result = result[my : result.shape[0] - my, mx : result.shape[1] - mx]

    cv2.imwrite(str(output_folder / path.name), result)
