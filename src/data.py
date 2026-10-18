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

def mask_to_regions(path, min_area=30, max_area=None, speck_size=10, gap_size=5):
    """Find each annotated barnacle by tracing its outline with cv2.findContours.
 
    The masks have two kinds of noise, handled in order:
      1. Floating specks: tiny clusters of line pixels that belong to no outline.
         Removed first, so they can't become fake barnacles or get glued onto
         real outlines by step 2.
      2. Small gaps in outlines: a morphological CLOSE (dilate, then erode)
         bridges gaps up to about gap_size pixels wide.
    Outlines whose gap is too big to bridge are repaired with their convex
    hull, which works because barnacles are roughly convex.
 
    How we know if an outline is closed: findContours with RETR_CCOMP returns
    a two-level hierarchy of outer boundaries and the holes inside them.
    A closed ring has a hole (its inside); an open "C" shape does not.
 
    Parameters (all judgment calls; tune them and say why in the notebook)
      min_area   : drop shapes smaller than this many pixels (after filling)
      max_area   : optionally drop implausibly large shapes, e.g. two outlines
                   that gap-closing accidentally merged
      speck_size : line-pixel clusters smaller than this are treated as noise
      gap_size   : width of gap (in pixels) that closing can bridge. Too big
                   and neighbouring outlines may merge into one shape.
 
    Returns
    -------
    regions  : (H, W) int array, 0 = not a barnacle, k = barnacle number k
    dots     : (N, 2) float array of barnacle centers in (x, y)
    areas    : (N,) float array of each barnacle's filled area in pixels
    repaired : (N,) bool array, True if the outline was open and the
               convex hull was used instead
    """
    lines = load_mask_lines(path).astype(np.uint8)
 
    # Step 1: remove floating specks. Each connected cluster of line pixels is
    # measured; clusters too small to be an outline are erased.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= speck_size
    keep[0] = False  # label 0 is the empty background
    lines = keep[labels].astype(np.uint8)
 
    # Step 2: bridge small gaps.
    if gap_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
        lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)
 
    # Step 3: trace outlines.
    contours, hierarchy = cv2.findContours(lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
 
    regions = np.zeros(lines.shape, dtype=np.int32)
    dots, areas, repaired = [], [], []
    if hierarchy is None:  # empty mask
        return regions, np.empty((0, 2)), np.array([]), np.array([], dtype=bool)
 
    # hierarchy[i] = [next, previous, first_child, parent]
    for contour, (_, _, child, parent) in zip(contours, hierarchy[0]):
        if parent != -1:
            continue  # this contour is a hole (the inside of a ring), skip it
        is_closed = child != -1
        shape = contour if is_closed else cv2.convexHull(contour)
 
        area = cv2.contourArea(shape)
        if area < min_area or (max_area is not None and area > max_area):
            continue
 
        m = cv2.moments(shape)  # centroid = (m10/m00, m01/m00)
        if m["m00"] == 0:
            continue
 
        label = len(dots) + 1
        cv2.drawContours(regions, [shape], -1, label, thickness=cv2.FILLED)
        dots.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        areas.append(area)
        repaired.append(not is_closed)
 
    return regions, np.array(dots, dtype=float), np.array(areas), np.array(repaired)
 
 
def mask_to_dots(path, min_area=30, max_area=None):
    """Ground-truth barnacle centers as an (N, 2) array of (x, y)."""
    _, dots, _, _ = mask_to_regions(path, min_area, max_area)
    return dots
 
 
def regions_to_filled_mask(regions, grow=0):
    """Boolean 'is this pixel barnacle?' mask, for training pixel classifiers.
 
    Filled contours already include the drawn outline, so no growing is
    needed by default. 'grow' is kept in case you want a safety margin.
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
    regions, dots, areas, repaired = mask_to_regions(f"mask{n}.png", min_area)
    if image.shape[:2] != regions.shape:
        raise ValueError(
            f"img{n} is {image.shape[:2]} but mask{n} is {regions.shape}; "
            "they must match pixel-for-pixel."
        )
    return {"image": image, "regions": regions, "dots": dots,
            "areas": areas, "repaired": repaired}
 
 
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
 