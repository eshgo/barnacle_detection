"""
data.py: loading images and turning the hand-annotated masks into ground truth.

  * load_image()      gives every other module images in one format (RGB uint8).
  * mask_to_regions() turns the scientists' contour masks into one region and
                      one (x, y) "dot" per barnacle. Dots are what we evaluate on.

What the annotations actually outline: looking at masked_img1/2, each blue
contour traces the barnacle's central plates (the operculum, the "diamond"),
not the whole white shell. So a "barnacle region" here means the operculum.

COORDINATE CONVENTION: every dot is (x, y) = (column, row), like OpenCV.
skimage.feature.peak_local_max returns (row, col); flip it before comparing.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# <repo>/data, found relative to this file so it works from any folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _resolve(path):
    """Accept a full path or just a filename inside data/."""
    path = Path(path)
    if not path.exists():
        path = DATA_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path):
    """Load a photo as RGB uint8 (H, W, 3). Drops any alpha channel.

    OpenCV loads BGR; matplotlib expects RGB. Converting once, here, means
    no other file has to think about it.
    """
    img = cv2.imread(str(_resolve(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def describe_file(path):
    """Print shape and per-channel value ranges. Look before assuming."""
    raw = cv2.imread(str(_resolve(path)), cv2.IMREAD_UNCHANGED)
    print(f"{Path(path).name}: shape={raw.shape}, dtype={raw.dtype}")
    channels = [raw] if raw.ndim == 2 else [raw[:, :, c] for c in range(raw.shape[2])]
    for c, ch in enumerate(channels):
        print(f"  channel {c}: min={ch.min()}, max={ch.max()}, unique={len(np.unique(ch))}")
    return raw


def load_mask_lines(path, threshold=0):
    """Boolean array, True where a contour line was drawn.

    The masks are RGBA with a transparent WHITE background, so color can't
    separate lines from background. Alpha can: alpha > 0 means "drawn".
    Falls back to "any nonzero pixel" for masks with no transparency.
    """
    raw = cv2.imread(str(_resolve(path)), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not decode {path}")
    if raw.ndim == 3 and raw.shape[2] == 4 and raw[:, :, 3].min() < 255:
        lines = raw[:, :, 3] > threshold
    else:
        gray = raw if raw.ndim == 2 else raw[:, :, :3].max(axis=2)
        lines = gray > threshold
    if lines.mean() > 0.5:
        raise ValueError(f"{lines.mean():.0%} of pixels look like lines; "
                         "the background is probably not transparent/black.")
    return lines


# ---------------------------------------------------------------------------
# Masks -> barnacle regions -> dots
# ---------------------------------------------------------------------------

def _disk(r):
    """Round structuring element of radius r."""
    r = max(0, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def estimate_line_width(lines):
    """Typical stroke thickness in pixels.

    The distance transform gives each line pixel its distance to the nearest
    non-line pixel; mid-stroke that's about (w + 1) / 2, so w = 2 * high - 1.
    Used PER CLUSTER below, because mask1 mixes thin strokes (~3 px) with a
    thick-stroke region (~8-10 px), so one global width was wrong for both.
    """
    dist = cv2.distanceTransform(lines.astype(np.uint8), cv2.DIST_L2, 3)
    values = dist[lines.astype(bool)]
    if values.size == 0:
        return 1.0
    return max(1.0, 2 * float(np.percentile(values, 95)) - 1)


def mask_to_regions(path, min_area=30, max_area=None, speck_size=10, gap_size=5,
                    min_hole_area=5, min_arc_size=15, merge_factor=1.8,
                    min_relative_area=0.15, return_details=False):
    """Find every annotated barnacle. Handles, in the real masks:

      floating pixels   clusters of line pixels < speck_size are noise
      gaps in outlines  closed per cluster; big gaps -> convex hull repair
      line "bubbles"    1-2 px holes inside a stroke (358 of 397 outline
                        clusters in mask2 have one!) are ignored: a real
                        barnacle's hole must be >= min_hole_area
      touching outlines two touching outlines are one cluster, but each
                        closed one encloses its own hole, so we count holes
      thick strokes     line width is measured per cluster, not globally
      filled outlines   some tiny barnacles are drawn so thick there's no hole;
                        they fall through to the hull repair, which is fine

    Per cluster of connected line pixels:
      a. cut it out on its own canvas (so neighbours can never fuse)
      b. every hole >= min_hole_area = one closed barnacle. Holes are found
         twice, with and without closing small gaps, keeping whichever finds
         more: closing rescues gapped outlines, but also fills the small
         interiors of tiny barnacles, so it's only used when it helps
      c. every stroke pixel is given to its NEAREST hole (if within about one
         line width). Body of a barnacle = its hole + its stroke pixels.
      d. stroke pixels nobody claimed = an outline that never closed -> hull.
         At most two outlines touch, so: 2 holes -> no open outline here;
         1 hole -> at most one open outline, so pool all leftover pieces;
         0 holes -> one open outline, or two touching ones (split in step e)
      e. a hull > merge_factor x the median closed barnacle is probably two
         touching open outlines: split it at its narrowest point
      f. drop shapes smaller than min_area pixels OR smaller than
         min_relative_area x the median barnacle (lens-shaped slivers etc.)

    Returns regions (H, W) int labels, dots (N, 2) (x, y), areas (N,),
    repaired (N,) bool, and optionally a details dict of counts.
    """
    lines = load_mask_lines(path)
    H, W = lines.shape
    close_kernel = _disk(gap_size // 2) if gap_size > 0 else None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8),
                                                           connectivity=8)
    details = {"specks": 0, "ignored_small_holes": 0, "gap_closed": 0,
               "closed": 0, "repaired": 0,
               "split_merged_arcs": 0, "dropped_overlapping_arcs": 0,
               "dropped_too_small": 0, "dropped_too_large": 0}
    candidates = []  # (local_mask, x0, y0, repaired, leftover_points_or_None)

    for i in range(1, n):  # label 0 = empty background
        x, y, w, h, size = stats[i]
        if size < speck_size:
            details["specks"] += 1
            continue

        piece_raw = labels[y:y + h, x:x + w] == i
        width = estimate_line_width(piece_raw)            # this cluster's own width
        reach = int(np.ceil(width)) + 1                    # how far a stroke extends
        pad = gap_size + reach + 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        piece = labels[y0:y1, x0:x1] == i

        # (a)+(b) find real holes, with and without gap closing, and keep the
        # version that finds MORE. Closing rescues gapped outlines but also
        # fills tiny barnacles' small interiors, so it's only used when it helps.
        raw = piece.astype(np.uint8)
        hole_ids, n_holes, n_bubbles = _find_holes(raw, min_hole_area)
        stroke_src = piece
        if close_kernel is not None:
            closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
            c_ids, c_n, c_bubbles = _find_holes(closed, min_hole_area)
            if c_n > n_holes:
                hole_ids, n_holes, n_bubbles = c_ids, c_n, c_bubbles
                stroke_src = piece | closed.astype(bool)
                details["gap_closed"] += 1
        details["ignored_small_holes"] += n_bubbles

        # (c) give each stroke pixel to its nearest hole
        claimed = np.zeros(piece.shape, bool)
        if n_holes:
            dist, (iy, ix) = ndimage.distance_transform_edt(hole_ids == 0, return_indices=True)
            nearest = hole_ids[iy, ix]                      # id of the closest hole
            stroke = stroke_src & (dist <= reach)
            for k in range(1, n_holes + 1):
                body = (hole_ids == k) | (stroke & (nearest == k))
                candidates.append((body, x0, y0, False, None))
                claimed |= body
            claimed = cv2.dilate(claimed.astype(np.uint8), _disk(1)).astype(bool)

        # (d) leftover stroke = open outline(s)
        if n_holes >= 2:
            continue                                        # both touching ones are closed
        leftover = (piece & ~claimed).astype(np.uint8)
        m, arc_labels, arc_stats, _ = cv2.connectedComponentsWithStats(leftover, connectivity=8)
        big = [j for j in range(1, m) if arc_stats[j, cv2.CC_STAT_AREA] >= min_arc_size]
        if not big:
            continue
        ys, xs = np.nonzero(np.isin(arc_labels, big))
        points = np.column_stack([xs, ys]).astype(np.int32)
        hull = _hull_mask(points, piece.shape)
        if (hull & claimed).sum() > 0.5 * hull.sum():
            details["dropped_overlapping_arcs"] += 1        # fragment of a found barnacle
            continue
        candidates.append((hull, x0, y0, True, points))

    # (e) split oversized hulls (two touching open outlines)
    closed_areas = [c[0].sum() for c in candidates if not c[3]]
    typical = float(np.median(closed_areas)) if closed_areas else None
    final = []
    for mask, x0, y0, rep, points in candidates:
        if rep and typical and mask.sum() > merge_factor * typical \
                and len(points) >= 2 * min_arc_size:
            details["split_merged_arcs"] += 1
            for half in _split_at_waist(points):
                final.append((_hull_mask(half, mask.shape), x0, y0, True))
        else:
            final.append((mask, x0, y0, rep))

    # (f) size filters, then paint regions and compute centers
    min_keep = max(min_area, min_relative_area * typical) if typical else min_area
    regions = np.zeros((H, W), dtype=np.int32)
    dots, areas, repaired = [], [], []
    for mask, x0, y0, rep in final:
        area = float(mask.sum())
        if area < min_keep:
            details["dropped_too_small"] += 1
            continue
        if max_area is not None and area > max_area:
            details["dropped_too_large"] += 1
            continue
        ys, xs = np.nonzero(mask)
        label = len(dots) + 1
        target = regions[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        target[mask & (target == 0)] = label                # never overwrite a neighbour
        dots.append((xs.mean() + x0, ys.mean() + y0))
        areas.append(area)
        repaired.append(rep)
        details["repaired" if rep else "closed"] += 1

    out = (regions, np.array(dots, dtype=float).reshape(-1, 2),
           np.array(areas), np.array(repaired, dtype=bool))
    return out + (details,) if return_details else out


def _find_holes(piece, min_hole_area):
    """Label the real holes of one outline cluster.

    Returns (hole_ids, n_holes, n_bubbles): hole_ids is an int image where
    each real hole has its own id 1..n. Holes smaller than min_hole_area are
    "bubbles" inside the stroke and are counted but not labeled.
    """
    contours, hierarchy = cv2.findContours(piece, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hole_ids = np.zeros(piece.shape, np.int32)
    n_holes = n_bubbles = 0
    if hierarchy is not None:
        for c, hier in zip(contours, hierarchy[0]):
            if hier[3] == -1:
                continue                                    # outer boundary
            if cv2.contourArea(c) < min_hole_area:
                n_bubbles += 1                              # bubble in the stroke
                continue
            n_holes += 1
            cv2.drawContours(hole_ids, [c], -1, n_holes, thickness=cv2.FILLED)
    return hole_ids, n_holes, n_bubbles


def _hull_mask(points, shape):
    """Filled convex hull of (x, y) points as a boolean mask."""
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 1)
    return mask.astype(bool)


def _split_at_waist(points):
    """Split the pixels of two touching outlines into two groups.

    Two touching rings form a dumbbell: long along the line joining their
    centers, and narrowest where they touch. So: find the long axis (PCA),
    project the pixels onto it, and cut where the fewest pixels are, searching
    only the middle 60% so we don't cut off a sliver at one end.
    (k-means was tried first; on thin 1-px rings it sometimes split diagonally,
    because rings aren't the compact blobs k-means assumes.)
    """
    pts = points.astype(float)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    lo, hi = np.percentile(proj, [20, 80])
    edges = np.arange(lo, hi + 2, 2.0)
    if len(edges) < 3:
        cut = np.median(proj)
    else:
        counts, _ = np.histogram(proj, bins=edges)
        k = int(np.argmin(counts))
        cut = (edges[k] + edges[k + 1]) / 2
    return [points[proj < cut], points[proj >= cut]]


def regions_to_filled_mask(regions, grow=0):
    """Boolean 'is this pixel barnacle?' mask, for training pixel classifiers."""
    filled = (regions > 0).astype(np.uint8)
    if grow > 0:
        filled = cv2.dilate(filled, _disk(grow))
    return filled.astype(bool)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def barnacle_stats(dots, areas):
    """Size and spacing statistics, used to justify parameters elsewhere.

    equivalent diameter -> kernel sizes, matching radius in evaluation
    nearest-neighbour distance -> min_distance when turning maps into dots
    """
    diameters = 2 * np.sqrt(np.asarray(areas) / np.pi)
    nn = cKDTree(dots).query(dots, k=2)[0][:, 1] if len(dots) > 1 else np.array([])
    return {"count": len(dots), "diameter": diameters, "nn_distance": nn}