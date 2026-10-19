"""
detect.py — turn an image into predicted barnacle dots.

Pipeline position:  crop + features  ->  [detect.py]  ->  predicted (x, y) dots

Two detectors, deliberately built to be compared:

  A. Baseline (no learning): a single blob kernel (LoG) at barnacle scale,
     then find its peaks. Every parameter is hand-picked.
  B. Random forest pixel classifier: learns from mask labels which feature
     combinations mean "barnacle", producing a probability map. Then the
     map is converted to dots with a distance transform + peak finding.

If B doesn't clearly beat A on held-out data, the extra complexity isn't
earning its keep, and that's a legitimate conclusion to report.

All returned dots are (x, y) = (column, row). peak_local_max returns
(row, col), so we flip it with [:, ::-1] every time.
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import peak_local_max

from .features import to_gray


# ---------------------------------------------------------------------------
# A. Baseline: blob kernel -> peaks
# ---------------------------------------------------------------------------

def log_score_map(image, radius):
    """Bright-blob score at one scale. High where a barnacle-sized bright blob is centered.

    A Laplacian of Gaussian responds most strongly to a blob of radius
    sigma*sqrt(2), so sigma = radius / sqrt(2). Use the median barnacle radius
    from data.barnacle_stats. Negated so bright blobs score positive;
    multiplied by sigma^2 so the scale doesn't change the score's magnitude.
    """
    sigma = radius / np.sqrt(2)
    smooth = cv2.GaussianBlur(to_gray(image), (0, 0), sigma)
    return -(sigma ** 2) * cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)


def peaks_to_dots(score, min_distance, threshold=None):
    """Local maxima of a score map -> (x, y) dots.

    min_distance: two peaks closer than this are treated as one barnacle.
                  Base it on nearest-neighbour spacing (data.barnacle_stats):
                  slightly less than the typical spacing.
    threshold:    ignore peaks with a score below this. Too low = dots on
                  random rock texture; too high = faint barnacles missed.
                  This is the main knob to tune on the TRAINING image.
    """
    coords = peak_local_max(score, min_distance=int(min_distance),
                            threshold_abs=threshold, exclude_border=False)
    return coords[:, ::-1].astype(float)


# ---------------------------------------------------------------------------
# B. Random forest pixel classifier
# ---------------------------------------------------------------------------

def sample_training_pixels(features, barnacle_mask, n_per_class=20000,
                           ignore_band=2, seed=0, valid=None):
    """Pick a balanced random sample of labeled pixels for training.

    Why sample instead of using every pixel: a 2000x2000 crop is 4 million
    pixels. Neighbouring pixels are nearly identical, so most of that is
    redundant, and the forest would train slowly for little gain.

    Why balanced: if 30% of pixels are barnacle, a lazy model can score 70%
    accuracy by predicting "rock" everywhere. Equal numbers per class force
    it to actually learn the difference.

    ignore_band: skip pixels within this many pixels of a barnacle edge.
    Edge labels are the least trustworthy (the hand-drawn outline is only
    roughly on the true edge), so leaving them out reduces label noise.
    Set to 0 to use everything.

    valid: optional boolean mask; only pixels where it's True are sampled.
    Used for spatial train/test splits (train on one half of an image).

    Returns X (n, F) features and y (n,) labels: 1 = barnacle, 0 = background.
    """
    fg = barnacle_mask.astype(np.uint8)
    if ignore_band > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * ignore_band + 1, 2 * ignore_band + 1))
        sure_fg = cv2.erode(fg, k).astype(bool)
        sure_bg = ~cv2.dilate(fg, k).astype(bool)
    else:
        sure_fg, sure_bg = fg.astype(bool), ~fg.astype(bool)
    if valid is not None:
        sure_fg, sure_bg = sure_fg & valid, sure_bg & valid

    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for label, region in [(1, sure_fg), (0, sure_bg)]:
        idx = np.flatnonzero(region)
        if len(idx) == 0:
            raise ValueError(f"No pixels available for class {label}.")
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        X_parts.append(features.reshape(-1, features.shape[-1])[chosen])
        y_parts.append(np.full(len(chosen), label))
    return np.concatenate(X_parts), np.concatenate(y_parts)


def train_pixel_classifier(X, y, n_estimators=100, min_samples_leaf=5, seed=0):
    """Fit a random forest on (features, label) pixel samples.

    Why a random forest: it handles features on different scales without
    normalization, captures non-linear rules ("bright AND round AND not
    green"), trains in seconds, and reports feature importances, so we can
    see WHICH kernels mattered. That last point is great for the writeup.

    min_samples_leaf=5 stops trees memorizing individual pixels, a mild guard
    against overfitting when all training data comes from one image.
    n_jobs=-1 uses all CPU cores.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    return clf


def predict_probability_map(clf, features, chunk_size=500_000, mask=None):
    """Barnacle probability for every pixel: (H, W) array in [0, 1].

    mask: optional boolean (H, W); only those pixels are predicted (the rest
    stay 0). Prediction is the slowest step, so skipping pixels we don't need
    (e.g. outside a test region) saves real time on a single CPU.
    Predicts in chunks so memory stays bounded on big images.
    """
    H, W, F = features.shape
    flat = features.reshape(-1, F)
    idx = np.arange(H * W) if mask is None else np.flatnonzero(mask)
    probs = np.zeros(H * W, dtype=np.float32)
    for start in range(0, len(idx), chunk_size):
        part = idx[start:start + chunk_size]
        probs[part] = clf.predict_proba(flat[part])[:, 1]
    return probs.reshape(H, W)


def probability_to_dots(prob, threshold=0.5, min_distance=5, min_blob_area=20, smooth=1.0):
    """Probability map -> one dot per barnacle. This is the "make them dots" step.

    1. Threshold: pixels with prob >= threshold are 'barnacle'.
    2. Drop blobs smaller than min_blob_area (isolated noisy pixels).
    3. Distance transform: each barnacle pixel's value = its distance to the
       nearest background pixel. A round blob becomes a hill whose peak is
       its center. Two barnacles merged by a thin bridge become two hills,
       because the bridge is thin (low distance) and both centers are far
       from the background (high distance). This is what separates
       touching barnacles.
    4. A slight blur flattens jagged plateaus so each hill has one peak.
    5. Each peak = one dot. Peaks are searched within each blob separately.

    Returns (dots, foreground_mask). The mask is useful for plotting and for
    count_by_area.
    """
    fg = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_blob_area
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    fg = (labels > 0).astype(np.uint8)

    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    if smooth > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), smooth)
    coords = peak_local_max(dist, min_distance=int(min_distance), labels=labels,
                            exclude_border=False)
    return coords[:, ::-1].astype(float), fg.astype(bool)


def count_by_area(foreground, typical_area):
    """Rough count = total barnacle area / area of a typical barnacle.

    A deliberately simple alternative that ignores individual barnacles.
    It's useful as a sanity check: if this and the dot count disagree wildly,
    something is off. Weakness: biased whenever barnacle sizes differ from
    the training image's typical size.
    """
    return float(np.sum(foreground)) / typical_area