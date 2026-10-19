"""
count_barnacles.py — count barnacles in field photos.

Usage:
    python count_barnacles.py                       # counts unseen_img1 + unseen_img2
    python count_barnacles.py photo1.png photo2.png # counts any photos in data/ or by path
    python count_barnacles.py --retrain             # ignore the cached model

What it does, for each photo:
  1. finds the green frame and crops to its biggest rectangle
     (no frame found -> uses the whole photo)
  2. predicts a barnacle probability for every pixel (random forest on
     kernel features), trained on BOTH labeled images (img1 + img2)
  3. turns the probability map into one dot per barnacle and counts them
  4. saves, in outputs/:
       <name>_detections.png   dots drawn on the crop, for a human to review
       <name>_probability.png  the probability map (where the model is unsure)
       counts.csv              one row per photo

The model is cached in models/barnacle_forest.joblib after the first run
(training takes ~30 s on one CPU core).
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data, pipeline

MODEL_PATH = Path("models/barnacle_forest.joblib")
OUT = Path("outputs")


def get_model(retrain=False):
    if MODEL_PATH.exists() and not retrain:
        print(f"Loading cached model from {MODEL_PATH}")
        return pipeline.load_model(MODEL_PATH)
    print("Training on img1 + img2 (all labeled pixels)...")
    labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
    for L in labeled:
        print(f"  {L['name']}: {len(L['gt'])} labeled barnacles, median radius {L['radius']:.1f} px")
    bundle = pipeline.train_model(labeled)
    pipeline.save_model(bundle, MODEL_PATH)
    return bundle


def save_review_images(name, result):
    """The detection overlay is the main output for a human reviewer: it's
    much faster to check dots and fix a few mistakes than to count 1000+."""
    crop, dots = result["crop"], result["dots"]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(crop)
    ax.scatter(dots[:, 0], dots[:, 1], s=8, c="red")
    ax.set_title(f"{name}: {result['count']} barnacles ({result['frame']['method']})")
    ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"{name}_detections.png", dpi=100); plt.close()

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(result["prob"], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"{name}: barnacle probability"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(OUT / f"{name}_probability.png", dpi=60); plt.close()


def consistency_check(results):
    """unseen_img2 is an exact 500x500 crop of unseen_img1 at x 1500-2000,
    y 1000-1500 (found by template matching; score 1.000 at scale 1.0).
    With no labels for the unseen photos, agreement between the two counts
    is the only check available: the same barnacles should be found twice."""
    if "unseen_img1" not in results or "unseen_img2" not in results:
        return
    raw = results["unseen_img1"]["dots_raw"]
    box = (raw[:, 0] >= 1500) & (raw[:, 0] < 2000) & (raw[:, 1] >= 1000) & (raw[:, 1] < 1500)
    in_box = int(box.sum())
    alone = results["unseen_img2"]["count"]
    print(f"\nConsistency check: unseen_img2 alone -> {alone} barnacles; the same "
          f"500x500 square inside unseen_img1 -> {in_box} barnacles "
          f"({100 * (alone - in_box) / max(in_box, 1):+.0f}% difference)")


def main(args):
    retrain = "--retrain" in args
    names = [a for a in args if not a.startswith("--")] or ["unseen_img1.png", "unseen_img2.png"]
    OUT.mkdir(exist_ok=True)
    bundle = get_model(retrain)

    results, rows = {}, []
    for path in names:
        name = Path(path).stem
        result = pipeline.count_image(bundle, data.load_image(path))
        results[name] = result
        save_review_images(name, result)
        print(f"{name}: {result['count']} barnacles  (frame: {result['frame']['method']}, "
              f"area counted: {result['frame']['shape'][1]}x{result['frame']['shape'][0]} px)")
        rows.append({"image": name, "count": result["count"], "frame": result["frame"]["method"]})

    with open(OUT / "counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "count", "frame"])
        writer.writeheader(); writer.writerows(rows)
    consistency_check(results)


if __name__ == "__main__":
    main(sys.argv[1:])
