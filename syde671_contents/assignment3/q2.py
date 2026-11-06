import cv2
import numpy as np

from q1 import computeH, warpImageBilinear, blendMosaic


def harris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    xx = cv2.GaussianBlur(x * x, (0, 0), 1.5)
    yy = cv2.GaussianBlur(y * y, (0, 0), 1.5)
    xy = cv2.GaussianBlur(x * y, (0, 0), 1.5)

    response = xx * yy - xy * xy - 0.04 * (xx + yy) ** 2
    response[:20] = response[-20:] = 0
    response[:, :20] = response[:, -20:] = 0

    peaks = response == cv2.dilate(response, np.ones((5, 5), np.uint8))
    y, x = np.where(peaks & (response > 0))
    scores = response[y, x]
    order = np.argsort(scores)[::-1]

    return np.column_stack((x[order], y[order])).astype(float), scores[order]


def anms(points, scores):
    points, scores = points[:3000], scores[:3000]
    radii = np.full(len(points), np.inf)

    for i in range(1, len(points)):
        stronger = scores[:i] > scores[i] / 0.9
        if stronger.any():
            radii[i] = ((points[:i][stronger] - points[i]) ** 2).sum(1).min()

    return points[np.argsort(radii)[::-1][:500]]


def descriptors(image, points):
    gray = (
        cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(
            np.float32
        )
        / 255
    )

    good_points, result = [], []

    for point in points:
        x, y = point.astype(int)
        if x < 20 or y < 20 or x >= gray.shape[1] - 20 or y >= gray.shape[0] - 20:
            continue

        patch = cv2.getRectSubPix(gray, (40, 40), tuple(point))
        patch = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
        patch = (patch - patch.mean()) / max(patch.std(), 1e-6)

        good_points.append(point)
        result.append(patch)

    return np.array(good_points), np.array(result)


def match(a, b):
    distance = ((a[:, None] - b[None, :]) ** 2).sum(2)
    a_to_b = np.argsort(distance, axis=1)[:, :2]
    b_to_a = np.argsort(distance, axis=0)[:2]

    matches = []
    for i, (best, second) in enumerate(a_to_b):
        forward_ratio = distance[i, best] / distance[i, second]
        backward_ratio = distance[i, best] / distance[b_to_a[1, best], best]

        if forward_ratio < 0.7 and b_to_a[0, best] == i and backward_ratio < 0.7:
            matches.append((i, best))

    return np.array(matches)


def ransac(source, destination):
    best = None

    for _ in range(5000):
        chosen = np.random.choice(len(source), 4, replace=False)
        h = computeH(source[chosen], destination[chosen])

        projected = h @ np.column_stack((source, np.ones(len(source)))).T
        projected = (projected[:2] / projected[2]).T

        inliers = np.linalg.norm(projected - destination, axis=1) < 3

        if best is None or inliers.sum() > best.sum():
            best = inliers

    return computeH(source[best], destination[best]), best


def mosaic_many(images, homographies):
    result = images[1]

    for i in [0, 2]:
        H = homographies[i]

        if i == 2:
            h, w = images[0].shape[:2]
            corners = (
                homographies[0]
                @ np.array(
                    [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]
                ).T
            )

            corners = corners[:2] / corners[2]
            min_x, min_y = np.floor(corners.min(axis=1)).astype(int)

            H = (
                np.array([[1, 0, -min(0, min_x)], [0, 1, -min(0, min_y)], [0, 0, 1]])
                @ H
            )

        warped = warpImageBilinear(images[i], H)
        mask = warpImageBilinear(np.ones(images[i].shape[:2]), H)
        result = blendMosaic(result, warped, mask, images[i].shape, H)

    return result


images = [
    cv2.imread("mosaic1.jpg"),
    cv2.imread("mosaic2.jpg"),
    cv2.imread("mosaic3.jpg"),
]

points = []
features = []
corner_counts = []

for image in images:
    corners, scores = harris(image)
    selected = anms(corners, scores)
    selected, feature = descriptors(image, selected)

    corner_counts.append(len(corners))
    points.append(selected)
    features.append(feature)


pair_12 = match(features[0], features[1])
h_2_to_1, inliers_12 = ransac(points[1][pair_12[:, 1]], points[0][pair_12[:, 0]])
h_1_to_2 = np.linalg.inv(h_2_to_1)

pair_23 = match(features[1], features[2])
h_3_to_2, inliers_23 = ransac(points[2][pair_23[:, 1]], points[1][pair_23[:, 0]])


manual_left = np.array(
    [
        [1182.3, 558.5],
        [1196.3, 657.1],
        [1248.8, 665.8],
        [1201.5, 623.8],
        [1247.0, 859.0],
        [1261.6, 704.3],
        [1256.4, 798.3],
        [1264.0, 683.3],
        [1210.9, 690.9],
        [1207.4, 727.1],
        [1192.8, 754.5],
        [1224.9, 606.3],
        [1218.4, 564.3],
        [1265.1, 641.9],
        [1161.3, 781.4],
    ]
)

manual_middle_left = np.array(
    [
        [25.1, 474.4],
        [42.0, 573.0],
        [94.5, 580.0],
        [47.9, 539.8],
        [92.2, 767.9],
        [105.6, 616.8],
        [100.4, 708.4],
        [109.1, 596.4],
        [56.6, 606.3],
        [53.1, 642.5],
        [39.1, 669.9],
        [69.4, 521.1],
        [63.6, 480.8],
        [110.3, 555.0],
        [5.3, 699.1],
    ]
)

manual_middle_right = np.array(
    [
        [973.9, 829.2],
        [1020.6, 806.5],
        [1055.6, 772.0],
        [1082.5, 900.4],
        [1034.6, 714.3],
        [1228.9, 809.4],
        [1265.7, 916.2],
        [1076.1, 522.3],
        [981.5, 632.6],
        [997.3, 899.2],
        [1258.1, 518.2],
        [1095.9, 491.9],
        [1031.1, 564.9],
        [1210.3, 522.3],
        [1225.4, 665.8],
    ]
)

manual_right = np.array(
    [
        [31.5, 736.4],
        [72.4, 712.5],
        [106.2, 675.2],
        [135.4, 798.3],
        [82.9, 618.0],
        [275.4, 697.9],
        [311.0, 796.0],
        [122.0, 423.7],
        [23.9, 537.4],
        [54.3, 805.3],
        [296.4, 416.7],
        [141.8, 392.7],
        [78.8, 467.4],
        [254.4, 420.7],
        [269.6, 560.8],
    ]
)


manual_h1 = computeH(manual_left, manual_middle_left)
manual_h3 = computeH(manual_right, manual_middle_right)

cv2.imwrite(
    "results/mosaic_three.png", mosaic_many(images, [h_1_to_2, np.eye(3), h_3_to_2])
)
cv2.imwrite(
    "results/mosaic_three_manual.png",
    mosaic_many(images, [manual_h1, np.eye(3), manual_h3]),
)
