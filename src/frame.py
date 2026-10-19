"""
frame.py — find the green wire frame and crop the photo to its inner square.

Pipeline position:  raw photo  ->  [frame.py]  ->  square crop of the counting area

Input : RGB photo (H, W, 3) containing the green frame.
Output: the inner square, perspective-corrected to a straight-on S x S image,
        plus the 3x3 transform M that maps raw-photo coordinates into the crop.

Why we keep M: ground truth (dots, regions) is defined in RAW photo coordinates.
To evaluate on the crop, the ground truth must go through the exact same
transform as the image. warp_dots() and warp_labels() do that.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: which pixels are green wire?
# ---------------------------------------------------------------------------

def green_mask(image, hue=(75, 110), min_sat=80, min_val=20, close_size=7):
    """Boolean mask of 'green wire' pixels.

    Why HSV and not RGB: in RGB, "green" depends on brightness (a shadowed green
    and a sunlit green have very different numbers). HSV separates color (Hue)
    from how vivid it is (Saturation) and how bright it is (Value), so one hue
    range catches the wire in both shadow and sun.

    OpenCV hue runs 0-179 (not 0-360). These thresholds were MEASURED, not
    guessed: sampling the wire on all four sides of img1, img2 and
    unseen_img1 gave hue 87-98 (teal, not pure green, which is ~60),
    saturation 125-200, and brightness as low as 31 in the shaded img1.
    My first guess, hue (35, 90) with min_val 40, missed most of the wire.
    min_sat=80 rejects dull wet rock and the paler green algae in img2.

    close_size: a morphological close fills small breaks in the wire caused by
    glare or a barnacle sitting on it. Without this, the inner square might not
    be fully enclosed and step 2 fails.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([hue[0], min_sat, min_val], dtype=np.uint8)
    upper = np.array([hue[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


# ---------------------------------------------------------------------------
# Step 2: find the four inner corners
# ---------------------------------------------------------------------------

def order_corners(points):
    """Sort 4 points into top-left, top-right, bottom-right, bottom-left.

    Trick: top-left has the smallest x+y and bottom-right the largest;
    top-right has the smallest y-x and bottom-left the largest. This works for
    any roughly upright quadrilateral, which a photographed frame is.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_from_contour(contour):
    """Simplify an outline to 4 corners.

    approxPolyDP replaces a wiggly outline with a polygon whose edges stay
    within epsilon pixels of it. We try increasing epsilon until we get
    exactly 4 corners. If that never happens (e.g. a corner is hidden), fall
    back to the minimum-area rotated rectangle around the outline.

    Returns (corners, method) so you can report how often the fallback fired.
    """
    perimeter = cv2.arcLength(contour, True)
    for eps in np.linspace(0.01, 0.10, 10):
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx), "polygon"
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(box), "min_area_rect"


def find_inner_corners(green):
    """Find the 4 corners of the frame's inner square from the green mask.

    Idea: the inner square is a HOLE in the green mask (a non-green region
    completely surrounded by green). findContours with RETR_CCOMP returns
    outer boundaries and holes separately, so we take the LARGEST hole.

    Why largest hole and not "largest green blob": a big patch of green algae
    could out-size the thin wire, but algae rarely encloses a huge empty
    region. The frame's interior is by far the biggest enclosed area.
    """
    contours, hierarchy = cv2.findContours(
        green.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        raise ValueError("No green found. Check the hue/saturation thresholds.")
    holes = [c for c, h in zip(contours, hierarchy[0]) if h[3] != -1]
    if not holes:
        raise ValueError(
            "Green found, but it doesn't enclose anything. The wire may be broken "
            "in the mask; try a larger close_size, or pick the corners by hand."
        )
    inner = max(holes, key=cv2.contourArea)
    return quad_from_contour(inner)


# ---------------------------------------------------------------------------
# Step 3: warp to a straight-on square
# ---------------------------------------------------------------------------

def warp_to_square(image, corners, size=None, interpolation=cv2.INTER_LINEAR):
    """Perspective-warp the quadrilateral at `corners` into a size x size square.

    Why warp instead of just cropping a bounding box: photos are rarely taken
    perfectly straight down, so the square frame appears as a skewed
    quadrilateral. A bounding-box crop would include rock outside the frame.
    A perspective warp maps the 4 corners exactly onto a square's corners.

    size defaults to the longest side, so we never shrink (lose detail).
    """
    corners = np.asarray(corners, dtype=np.float32)
    if size is None:
        sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        size = int(round(max(sides)))
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                      dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(image, M, (size, size), flags=interpolation)
    return warped, M, size


def crop_to_frame(image, min_frame_fraction=0.05, **green_kwargs):
    """Full frame step: green mask -> inner corners -> warped square.

    If no frame is found (no teal, nothing enclosed, or the largest enclosed
    region is under min_frame_fraction of the image, e.g. a speck of teal in
    a close-up photo like unseen_img2), the WHOLE image is used instead, with
    an identity transform, and method = 'no_frame'. Counting the whole
    picture is the sensible default when there's no frame to restrict it.

    Returns a dict:
      crop    : counting area (RGB)
      M       : 3x3 transform, raw-photo coords -> crop coords
      shape   : (height, width) of the crop
      size    : crop side length (max of shape), kept for convenience
      corners : (4, 2) inner corners in the raw photo (TL, TR, BR, BL)
      method  : 'polygon', 'min_area_rect', or 'no_frame'
    """
    H, W = image.shape[:2]
    try:
        green = green_mask(image, **green_kwargs)
        corners, method = find_inner_corners(green)
        if cv2.contourArea(corners) < min_frame_fraction * H * W:
            raise ValueError("largest enclosed region is too small to be the frame")
    except ValueError:
        corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        return {"crop": image.copy(), "M": np.eye(3), "shape": (H, W), "size": max(H, W),
                "corners": corners, "method": "no_frame"}
    crop, M, size = warp_to_square(image, corners)
    return {"crop": crop, "M": M, "shape": (size, size), "size": size,
            "corners": corners, "method": method}


def crop_with_manual_corners(image, corners):
    """Fallback when automatic detection fails: you supply the 4 corners.

    In a real tool this would be four clicks by the scientist, which is
    still far faster than counting barnacles.
    """
    corners = order_corners(corners)
    crop, M, size = warp_to_square(image, corners)
    return {"crop": crop, "M": M, "shape": (size, size), "size": size,
            "corners": corners, "method": "manual"}


# ---------------------------------------------------------------------------
# Moving ground truth into crop coordinates
# ---------------------------------------------------------------------------

def _hw(shape_or_size):
    """Accept an int side length or an (height, width) pair."""
    if np.isscalar(shape_or_size):
        return int(shape_or_size), int(shape_or_size)
    return int(shape_or_size[0]), int(shape_or_size[1])


def warp_dots(dots, M, shape):
    """Transform (x, y) dots into crop coordinates; drop dots outside the crop.

    Dropping is correct, not a hack: the scientists only count barnacles inside
    the inner square, so dots outside it aren't part of the ground truth.
    shape: the crop's (height, width), or a single side length.
    """
    h, w = _hw(shape)
    dots = np.asarray(dots, dtype=np.float32).reshape(-1, 2)
    if len(dots) == 0:
        return dots.astype(float)
    moved = cv2.perspectiveTransform(dots.reshape(-1, 1, 2), np.asarray(M, np.float64)).reshape(-1, 2)
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < w) & (moved[:, 1] >= 0) & (moved[:, 1] < h)
    return moved[inside].astype(float)


def warp_labels(labels, M, shape):
    """Warp a label image (e.g. regions from data.py) into crop coordinates.

    INTER_NEAREST is essential: normal interpolation would blend barnacle #12
    and #13 into a nonexistent barnacle #12.5. Nearest-neighbour only ever
    copies existing labels. (Converted to float32 because warpPerspective
    doesn't accept int32 images.)
    """
    h, w = _hw(shape)
    warped = cv2.warpPerspective(labels.astype(np.float32), np.asarray(M, np.float64), (w, h),
                                 flags=cv2.INTER_NEAREST)
    return warped.astype(labels.dtype)


def unwarp_dots(dots, M):
    """Map (x, y) dots from crop coordinates back to raw-photo coordinates.

    The inverse of warp_dots, using the inverse matrix of M. Needed to draw
    detections on the original photo, or to compare two photos of the same
    scene (unseen_img2 is an exact crop of unseen_img1).
    """
    dots = np.asarray(dots, dtype=np.float32).reshape(-1, 2)
    if len(dots) == 0:
        return dots.astype(float)
    Minv = np.linalg.inv(np.asarray(M, np.float64))
    return cv2.perspectiveTransform(dots.reshape(-1, 1, 2), Minv).reshape(-1, 2).astype(float)