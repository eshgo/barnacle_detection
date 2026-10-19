"""
pipeline.py — the whole system, end to end, in one place.

Both scripts use these functions:
  run_experiments.py   measures how well the method works (held-out data)
  count_barnacles.py   trains the final model and counts unseen photos

Keeping one shared code path matters: the numbers reported in the log then
describe exactly the program a scientist would run.

Steps for a new photo:
  photo -> frame.crop_to_frame -> features.compute_features
        -> random forest probability map -> detect.probability_to_dots -> count
"""

from pathlib import Path

import joblib
import numpy as np

from . import data, detect, features, frame

# ---------------------------------------------------------------------------
# Labeled data
# ---------------------------------------------------------------------------

def load_labeled(n):
    """Load img{n} + mask{n}, crop to the frame, move ground truth into the crop.

    Returns a dict with the crop, ground-truth dots (crop coords), a boolean
    barnacle-pixel mask (crop coords), and size statistics.
    """
    image = data.load_image(f"img{n}.png")
    fr = frame.crop_to_frame(image)
    regions, dots, areas, repaired = data.mask_to_regions(f"mask{n}.png")
    stats = data.barnacle_stats(dots, areas)
    return {
        "name": f"img{n}",
        "crop": fr["crop"],
        "frame": fr,
        "gt": frame.warp_dots(dots, fr["M"], fr["shape"]),
        "fg": data.regions_to_filled_mask(frame.warp_labels(regions, fr["M"], fr["shape"])),
        "radius": float(np.median(stats["diameter"]) / 2),
        "spacing": float(np.median(stats["nn_distance"])),
        "n_repaired": int(repaired.sum()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(labeled, valid=None, feats=None, n_per_class=20000, seed=0):
    """Train the pixel classifier on one or more labeled crops.

    labeled : list of dicts from load_labeled
    valid   : optional list of boolean masks (one per crop); only those pixels
              are used for training (for spatial train/test splits)
    feats   : optional precomputed feature stacks (saves time in experiments)

    Also fixes the dot-extraction parameters from the TRAINING data only:
      min_distance = smallest typical barnacle radius among training images
                     (two dots closer than one small-barnacle radius are
                     almost certainly the same barnacle)
      min_blob_area = 15% of a small barnacle's area, the same relative
                     threshold that separated junk from barnacles in the masks
    Returns a "bundle" dict: everything needed to run on a new photo.
    """
    if feats is None:
        feats = [features.compute_features(L["crop"])[0] for L in labeled]
    Xs, ys = [], []
    for k, L in enumerate(labeled):
        X, y = detect.sample_training_pixels(
            feats[k], L["fg"], n_per_class=n_per_class, seed=seed + k,
            valid=None if valid is None else valid[k])
        Xs.append(X); ys.append(y)
    clf = detect.train_pixel_classifier(np.concatenate(Xs), np.concatenate(ys), seed=seed)
    r_small = min(L["radius"] for L in labeled)
    return {
        "clf": clf,
        "feature_names": features.compute_features(labeled[0]["crop"][:40, :40])[1],
        "threshold": 0.5,
        "min_distance": r_small,
        "min_blob_area": int(0.15 * np.pi * r_small ** 2),
        "trained_on": [L["name"] for L in labeled],
    }


def save_model(bundle, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_model(path):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_in_crop(bundle, crop, feats=None, mask=None):
    """Probability map + dots for one crop. mask limits where we predict."""
    if feats is None:
        feats = features.compute_features(crop)[0]
    prob = detect.predict_probability_map(bundle["clf"], feats, mask=mask)
    dots, fg = detect.probability_to_dots(prob, bundle["threshold"],
                                          min_distance=bundle["min_distance"],
                                          min_blob_area=bundle["min_blob_area"])
    return dots, prob, fg


def count_image(bundle, image):
    """Full system on one raw photo. Returns a dict with:
      count, dots (crop coords), dots_raw (photo coords), crop, prob, frame
    """
    fr = frame.crop_to_frame(image)
    dots, prob, fg = detect_in_crop(bundle, fr["crop"])
    return {
        "count": len(dots),
        "dots": dots,
        "dots_raw": frame.unwarp_dots(dots, fr["M"]),
        "crop": fr["crop"],
        "prob": prob,
        "frame": fr,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def half_mask(shape, side, buffer=60):
    """Left ('L') or right ('R') half of an image, minus a buffer strip.

    The buffer keeps train and test pixels apart: the largest kernel
    (sigma 16) looks about 3 sigma = 48 px around each pixel, so without a
    gap, test pixels near the middle would share context with training ones.
    """
    h, w = shape
    m = np.zeros((h, w), bool)
    mid = w // 2
    if side == "L":
        m[:, :max(0, mid - buffer)] = True
    else:
        m[:, mid + buffer:] = True
    return m


def dots_inside(dots, mask):
    """Keep only dots that fall where mask is True."""
    dots = np.asarray(dots).reshape(-1, 2)
    if len(dots) == 0:
        return dots
    xi = np.clip(dots[:, 0].astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(dots[:, 1].astype(int), 0, mask.shape[0] - 1)
    return dots[mask[yi, xi]]
