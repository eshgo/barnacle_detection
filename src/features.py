"""
features.py — kernels that describe each pixel's neighbourhood.

Pipeline position:  crop  ->  [features.py]  ->  per-pixel feature stack

A single pixel's color says almost nothing ("is this light-gray pixel part of
a barnacle or a pale rock?"). Its NEIGHBOURHOOD says a lot: is it inside a
round bright blob of barnacle size? Near a strong edge? Each kernel below
answers one such question, at several scales. The random forest in detect.py
then learns which combination of answers means "barnacle".

This mirrors how the tool ilastik does pixel classification.
"""

import cv2
import numpy as np


def to_gray(image):
    """RGB uint8 -> grayscale float32 in [0, 1].

    Floats instead of uint8 because filters produce negative values and
    fractions (an edge filter's output can be -0.3), which uint8 can't hold.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def compute_features(image, sigmas=(1, 2, 4, 8, 16)):
    """Stack of filter responses: (H, W, F) array plus a name for each feature.

    For each scale sigma (in pixels) we compute three kernels:
      smooth_s   Gaussian blur. "How bright is this neighbourhood on average?"
                 Removes texture/noise smaller than sigma.
      grad_s     Gradient magnitude of the blurred image (Sobel kernels).
                 "Is there an edge here?" Barnacle rims have strong edges.
                 Multiplied by sigma so different scales are comparable.
      log_s      Negative Laplacian of Gaussian, times sigma^2.
                 "Is this the center of a bright blob about sigma*sqrt(2)
                 pixels in radius?" This is the classic blob detector.
                 Positive = bright blob, negative = dark blob.

    Why several sigmas (1, 2, 4, 8, 16): barnacles vary in size, and we don't know in advance
    which scale is informative. Small sigmas see shell texture; large sigmas
    see whole barnacles and the gaps between them. The random forest picks.
    The sigmas span ~1 px up to beyond the largest typical barnacle radius:
    img1's opercula have a median radius of ~10 px, img2's ~19 px, so 16 was
    added after the first real-data run (a LoG at sigma 16 peaks for blobs of
    radius ~23 px). log_16 then became one of the most important features.

    Plus two color features: the 'a' (green-red) and 'b' (blue-yellow)
    channels of Lab color space, lightly blurred. Barnacle shells are pale and
    grayish; rock, algae, and shadows differ in tint. Lab separates tint from
    brightness better than RGB.
    """
    gray = to_gray(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

    features, names = [], []
    for s in sigmas:
        smooth = cv2.GaussianBlur(gray, (0, 0), s)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad = s * np.sqrt(gx ** 2 + gy ** 2)
        log = -(s ** 2) * cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
        features += [smooth, grad, log]
        names += [f"smooth_{s}", f"grad_{s}", f"log_{s}"]

    for channel, name in [(1, "lab_a"), (2, "lab_b")]:
        features.append(cv2.GaussianBlur(lab[:, :, channel], (0, 0), 2))
        names.append(name)

    return np.stack(features, axis=-1).astype(np.float32), names


# ---------------------------------------------------------------------------
# A kernel learned from the data: the "average barnacle"
# ---------------------------------------------------------------------------

def make_template(image, dots, size):
    """Average the grayscale patches around every labeled barnacle.

    The result is a size x size kernel that looks like a typical barnacle.
    Then we make it rotation-symmetric by averaging it with its 90-degree
    rotations and mirror images, because barnacles have no preferred
    orientation, and we don't want the kernel to favour one.

    size should be odd and a bit larger than the typical barnacle diameter
    (~1.5x), so the kernel includes some surrounding rock: "a bright blob
    WITH darker surroundings" is more distinctive than the blob alone.
    """
    if size % 2 == 0:
        size += 1  # odd size so the template has a true center pixel
    gray = to_gray(image)
    H, W = gray.shape
    half = size // 2
    patches = []
    for x, y in np.round(np.asarray(dots)).astype(int):
        if half <= x < W - half and half <= y < H - half:
            patches.append(gray[y - half:y + half + 1, x - half:x + half + 1])
    if not patches:
        raise ValueError("No dots far enough from the edge to cut a patch.")
    template = np.mean(patches, axis=0)
    variants = [np.rot90(template, k) for k in range(4)]
    variants += [np.fliplr(v) for v in variants]
    return np.mean(variants, axis=0).astype(np.float32)


def template_response(image, template):
    """Slide the template over the image; return a same-size similarity map.

    TM_CCOEFF_NORMED = normalized correlation, from -1 to 1. 'Normalized'
    means it compares the SHAPE of the brightness pattern, not absolute
    brightness, so the same barnacle in shade or sun scores similarly.

    matchTemplate's output is smaller than the image (the template can't hang
    off the edge), so we pad it back to full size. Padding with 0 = "no
    evidence either way" at the borders.
    """
    gray = to_gray(image)
    response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape
    top, left = h // 2, w // 2
    return cv2.copyMakeBorder(response, top, h - 1 - top, left, w - 1 - left,
                              cv2.BORDER_CONSTANT, value=0)