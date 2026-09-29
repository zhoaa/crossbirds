import cv2
import numpy as np
from pathlib import Path


def ncc(A, B):
    # A, B are grayscale images (matrices) -> [-1, 1]
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
    best_score = -1
    best_shift = (0, 0)
    radius = 50

    # check every shift within the search radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            y0 = max(0, dy)
            y1 = min(anchor.shape[0], anchor.shape[0] + dy)

            x0 = max(0, dx)
            x1 = min(anchor.shape[1], anchor.shape[1] + dx)

            # compare only the overlapping regions
            a = anchor[y0:y1, x0:x1]
            b = cand[y0 - dy:y1 - dy, x0 - dx:x1 - dx]

            score = ncc(a, b)

            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    return best_score, best_shift


input_folder = Path(r"/Users/aaronzhou/development/w27-school/syde671/data/data 2")
output_folder = Path(r"/Users/aaronzhou/development/w27-school/syde671/outpput5")

for path in sorted(input_folder.iterdir()):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    h = image.shape[0]
    third = h // 3

    # Three vertically stacked exposures: B, G, R
    B = image[:third]
    G = image[third:2 * third]
    R = image[2 * third:3 * third]

    # take off a twelfth of each image from each side
    crop = B.shape[0] // 12

    crop = third // 12

    B_cropped = B[crop:-crop, crop:-crop]
    G_cropped = G[crop:-crop, crop:-crop]
    R_cropped = R[crop:-crop, crop:-crop]

    h = min(B_cropped.shape[0], G_cropped.shape[0], R_cropped.shape[0])
    w = min(B_cropped.shape[1], G_cropped.shape[1], R_cropped.shape[1])

    B_cropped = B_cropped[:h, :w]
    G_cropped = G_cropped[:h, :w]
    R_cropped = R_cropped[:h, :w]

    # shrink images by a factor of 4 for faster alignment
    shrink = 4

    B_small = B_cropped[::shrink, ::shrink]
    G_small = G_cropped[::shrink, ::shrink]
    R_small = R_cropped[::shrink, ::shrink]

    g_ncc, (gdx, gdy) = align(B_small, G_small)
    r_ncc, (rdx, rdy) = align(B_small, R_small)

    print("Image:", path.name)
    print("G offset (shrunk):", str((gdx, gdy)) + ", R offset (shrunk):", str((rdx, rdy)))

    # convert offsets back to the original image scale
    gdx *= shrink
    gdy *= shrink

    rdx *= shrink
    rdy *= shrink

    print("G offset:", str((gdx, gdy)) + ", R offset:", str((rdx, rdy)))

    # align the uncropped images
    G = np.roll(G, (gdy, gdx), axis=(0, 1))
    R = np.roll(R, (rdy, rdx), axis=(0, 1))

    result = cv2.merge([B, G, R])

    cv2.imwrite(str(output_folder / path.name), result)
