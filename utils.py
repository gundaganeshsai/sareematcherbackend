import cv2
import numpy as np
import random

def extract_colors_with_positions(image_bytes, k=5, sample_points=30):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return []

    h, w, _ = img.shape
    img_small = cv2.resize(img, (300, 300))

    data = img_small.reshape((-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    colors = []
    for i, c in enumerate(centers):
        hex_color = "#%02x%02x%02x" % (c[2], c[1], c[0])  # BGR→RGB
        positions = []
        mask = (labels.flatten() == i).reshape(img_small.shape[:2])
        ys, xs = np.where(mask)

        if len(xs) > 0:
            sample_idx = random.sample(range(len(xs)), min(sample_points, len(xs)))
            positions = [{"x": int(xs[j] * w / 300), "y": int(ys[j] * h / 300)} for j in sample_idx]

        colors.append({"hex": hex_color, "positions": positions})

    return colors
