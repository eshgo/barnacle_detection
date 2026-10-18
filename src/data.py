"""
data.py — loading images and turning the hand-annotated masks into ground truth.

Everything else in the project depends on this file:
  * load_image()   gives every other module images in one consistent format.
  * mask_to_dots() turns the scientists' contour masks into ground-truth "dots"
                   (one (x, y) point per barnacle), which is what we evaluate against.

COORDINATE CONVENTION (important!):
  All dots in this project are stored as (x, y) = (column, row), matching OpenCV.
  Some libraries (e.g. skimage.feature.peak_local_max) return (row, col) instead.
  Always convert to (x, y) before comparing dots.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

# images live in barnacle_detection/data/.
# Building the path from this file's location means the code works no matter which folder you launch the notebook from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _resolve(path):
    """Let callers pass either a full path or just a filename like 'img1.png'."""
    path = Path(path)
    if not path.exists():
        path = DATA_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path


# Loading image

def load_image(path):
    """Load a photo as an RGB uint8 array of shape (H, W, 3).

    OpenCV loads images in BGR order; matplotlib expects RGB. Converting here,
    once, means no other file ever has to think about it.
    """
    img = cv2.imread(str(_resolve(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def describe_file(path):
    """Print what a file actually contains (shape, channels, value ranges).

    Use this before trusting any assumption about the masks: are they
    transparent PNGs? Black backgrounds? Colored lines? Look first.
    """
    raw = cv2.imread(str(_resolve(path)), cv2.IMREAD_UNCHANGED)
    print(f"{Path(path).name}: shape={raw.shape}, dtype={raw.dtype}")
    channels = [raw] if raw.ndim == 2 else [raw[:, :, c] for c in range(raw.shape[2])]
    for c, ch in enumerate(channels):
        print(f"  channel {c}: min={ch.min()}, max={ch.max()}, "
              f"unique values={len(np.unique(ch))}")
    return raw


def load_mask_lines(path, threshold=0):
    """Return a boolean array that is True wherever a contour line was drawn.

    Handles countour with RGBA with a transparent background. the alpha channel marks the lines.
    """
    raw = cv2.imread(str(_resolve(path)), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not decode {path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        lines = raw[:, :, 3] > threshold          # non line pixels are transparent

    return lines


# Converting masks to easily countable barnacle region dots

def mask_to_regions(path, min_area=30, max_area=None):
    """Find each barnacle as the region enclosed by its contour line.

    Instead of tracing the drawn lines, look at the NON-line pixels. Every closed contour
    encloses its own separate pocket of non-line pixels, even if its outline
    touches a neighbour's. So each enclosed pocket = one barnacle.

    Filters applied (each is a judgment call worth discussing in the notebook):
      * Pockets touching the image border are the open background, not barnacles.
      * Pockets smaller than min_area are usually gaps squeezed between three
        touching outlines, not real barnacles. Tune this from the area histogram.
      * Optional max_area drops anything implausibly large.

    Known limitation: a contour that isn't fully closed doesn't enclose a
    pocket, so that barnacle is silently lost. Compare your count with the
    scientists' expectation to see how often this happens.

    Returns
    -------
    regions : (H, W) int array, 0 = not a barnacle, k = barnacle number k
    dots    : (N, 2) float array of barnacle centers in (x, y)
    areas   : (N,) int array of each barnacle's interior area in pixels
    """
    lines = load_mask_lines(path)
    h, w = lines.shape

    # connectivity=4 matters: with 8-connectivity, background could "leak"
    # diagonally through thin lines and merge pockets that should be separate.
    pockets = (~lines).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(pockets, connectivity=4)

    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]

    touches_border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    keep = ~touches_border & (area >= min_area)
    if max_area is not None:
        keep &= area <= max_area
    keep[0] = False  # label 0 is the line pixels themselves

    # Renumber the kept pockets 1..N so the region image is easy to use later.
    remap = np.zeros(n, dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    regions = remap[labels]

    return regions, centroids[keep].astype(float), area[keep]


def mask_to_dots(path, min_area=30, max_area=None):
    """Ground-truth barnacle centers as an (N, 2) array of (x, y)."""
    _, dots, _ = mask_to_regions(path, min_area, max_area)
    return dots


def regions_to_filled_mask(regions, grow=2):
    """Boolean 'is this pixel barnacle?' mask, for training pixel classifiers.

    The pockets exclude the drawn outline itself, so they're slightly smaller
    than the real barnacle. Dilating by roughly the line thickness ('grow')
    adds that rim back. This is an approximation, and worth saying so.
    """
    filled = (regions > 0).astype(np.uint8)
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        filled = cv2.dilate(filled, kernel)
    return filled.astype(bool)


# ---------------------------------------------------------------------------
# Convenience + exploration
# ---------------------------------------------------------------------------

def load_example(n, min_area=30):
    """Load img{n}.png with its ground truth. Checks that image and mask align."""
    image = load_image(f"img{n}.png")
    regions, dots, areas = mask_to_regions(f"mask{n}.png", min_area)
    if image.shape[:2] != regions.shape:
        raise ValueError(
            f"img{n} is {image.shape[:2]} but mask{n} is {regions.shape}; "
            "they must match pixel-for-pixel."
        )
    return {"image": image, "regions": regions, "dots": dots, "areas": areas}


def barnacle_stats(dots, areas):
    """Size and spacing statistics, used to justify parameters later.

    * equivalent diameter: diameter of a circle with the same area.
      -> informs blob/kernel sizes and the matching radius in evaluation.
    * nearest-neighbour distance: how tightly packed barnacles are.
      -> informs min_distance when finding peaks (dots) in predictions.
    """
    diameters = 2 * np.sqrt(np.asarray(areas) / np.pi)
    nn_dist = cKDTree(dots).query(dots, k=2)[0][:, 1] if len(dots) > 1 else np.array([])
    return {"count": len(dots), "diameter": diameters, "nn_distance": nn_dist}