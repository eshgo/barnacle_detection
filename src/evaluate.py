"""
evaluate.py — how good are the predicted dots?

Pipeline position:  predicted dots + ground-truth dots  ->  [evaluate.py]  ->  metrics, plots

Two levels of evaluation, because they answer different questions:
  * COUNT metrics: is the total right? This is what scientists record.
  * DETECTION metrics (precision / recall / F1): are the right barnacles
    found? A model can get the count right by luck, with misses and false
    alarms cancelling out. Detection metrics catch that.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_dots(pred, true, radius):
    """Pair predicted dots with true dots one-to-one, within `radius` pixels.

    Why one-to-one: without it, one prediction sitting between two barnacles
    could "claim" both and inflate the score. Each true barnacle can be found
    at most once, and each prediction can find at most one barnacle.

    Why the Hungarian algorithm (linear_sum_assignment): it finds the pairing
    with the smallest total distance. Pairs farther apart than `radius` get a
    huge cost, so the algorithm avoids them whenever possible, and we discard
    any that remain. A greedy "closest pair first" approach usually gives the
    same answer, but can mis-pair in dense clusters.

    radius: how far off a prediction can be and still count as a hit.
    A sensible choice is about half the typical barnacle diameter: close
    enough that the dot is clearly on THAT barnacle.

    Returns a dict of metrics plus the matched index pairs (for plotting).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    true = np.asarray(true, dtype=float).reshape(-1, 2)
    pred_idx = true_idx = np.array([], dtype=int)

    if len(pred) and len(true):
        dist = cdist(pred, true)  # (n_pred, n_true) distance matrix
        cost = np.where(dist <= radius, dist, 1e6)
        rows, cols = linear_sum_assignment(cost)
        ok = dist[rows, cols] <= radius
        pred_idx, true_idx = rows[ok], cols[ok]

    tp = len(pred_idx)
    fp = len(pred) - tp  # predictions matching no barnacle
    fn = len(true) - tp  # barnacles no prediction found
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(true) if len(true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_error = len(pred) - len(true)  # signed: + overcount, - undercount

    return {
        "n_pred": len(pred), "n_true": len(true),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "count_error": count_error,
        "pct_count_error": 100 * count_error / len(true) if len(true) else float("nan"),
        "pred_idx": pred_idx, "true_idx": true_idx,
    }


def summarize(result, name=""):
    """One readable line of the metrics that matter."""
    r = result
    print(f"{name:>14} | count {r['n_pred']:>5} vs {r['n_true']:>5} "
          f"({r['pct_count_error']:+.1f}%) | P {r['precision']:.2f}  "
          f"R {r['recall']:.2f}  F1 {r['f1']:.2f}")


def tile_counts(pred, true, size, n_tiles=4):
    """Count predicted and true dots in an n_tiles x n_tiles grid over the crop.

    Why: one overall number hides WHERE errors happen. Maybe the model is
    fine in sparse areas but undercounts dense clusters, or fails near the
    shadowed edge. Tile errors show that, and they suggest which tiles a human
    should double-check, which feeds the human-in-the-loop design.

    Returns (pred_grid, true_grid), each (n_tiles, n_tiles). Row 0 = top.
    """
    edges = np.linspace(0, size, n_tiles + 1)

    def grid(dots):
        dots = np.asarray(dots).reshape(-1, 2)
        g, _, _ = np.histogram2d(dots[:, 1], dots[:, 0], bins=[edges, edges])
        return g.astype(int)

    return grid(pred), grid(true)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_overlay(image, pred, true, result, ax=None, title=None, zoom=None):
    """Show hits, false alarms, and misses on the image.

    Shapes AND colors differ, so the plot still reads for colorblind viewers
    and in grayscale:
      hit (matched prediction)  green filled dot
      false alarm               red x
      miss (unfound barnacle)   blue hollow circle
    """
    pred = np.asarray(pred).reshape(-1, 2)
    true = np.asarray(true).reshape(-1, 2)
    hit = np.zeros(len(pred), bool); hit[result["pred_idx"]] = True
    found = np.zeros(len(true), bool); found[result["true_idx"]] = True

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.scatter(pred[hit, 0], pred[hit, 1], s=10, c="lime", label=f"hit ({hit.sum()})")
    ax.scatter(pred[~hit, 0], pred[~hit, 1], s=25, c="red", marker="x",
               label=f"false alarm ({(~hit).sum()})")
    ax.scatter(true[~found, 0], true[~found, 1], s=40, facecolors="none",
               edgecolors="deepskyblue", linewidths=1.5, label=f"miss ({(~found).sum()})")
    if zoom is not None:
        x0, x1, y0, y1 = zoom
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(title or f"F1 = {result['f1']:.2f}")
    ax.legend(loc="upper right")
    ax.axis("off")
    return ax


def plot_tile_errors(pred_grid, true_grid, ax=None, title="Count error per tile (pred - true)"):
    """Heatmap of per-tile count error. Red = overcount, blue = undercount."""
    err = pred_grid - true_grid
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lim = max(1, np.abs(err).max())
    im = ax.imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(err):
        ax.text(j, i, f"{v:+d}\n({true_grid[i, j]})", ha="center", va="center", fontsize=8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_mask_dots(lines, regions, dots, repaired, ax=None, zoom=None, title=None):
    """Ground-truth sanity plot: mask outlines + extracted dots (from data.py).

    white = background, light blue = counted barnacle area, black = outline,
    magenta = outline pixels that were thrown away (specks / filtered shapes),
    red dot = closed outline, orange dot = repaired with convex hull.
    """
    canvas = np.ones((*lines.shape, 3))
    canvas[regions > 0] = [0.80, 0.90, 1.00]
    canvas[lines] = [0, 0, 0]
    canvas[lines & (regions == 0)] = [1, 0, 1]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(canvas, interpolation="nearest")
    ax.scatter(dots[~repaired, 0], dots[~repaired, 1], s=12, c="red", label="closed outline")
    ax.scatter(dots[repaired, 0], dots[repaired, 1], s=12, c="orange", label="repaired (hull)")
    if zoom is not None:
        x0, x1, y0, y1 = zoom
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(title or f"{len(dots)} dots, {repaired.sum()} repaired")
    ax.legend(loc="upper right")
    return ax