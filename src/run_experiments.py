"""
run_experiments.py — measure how well the barnacle counter works.

Usage:
    python run_experiments.py split    # main result: train on one half of both
                                       # labeled images, test on the other half
    python run_experiments.py cross    # train on one image, test on the other

Results are printed and saved to outputs/metrics_<experiment>.txt, with
figures in figures/.

Why two experiments: they answer different questions.
  split: "new photo that looks like the training photos" (the unseen photos
         resemble img1, so this is the relevant estimate)
  cross: "new photo that looks NOTHING like any training photo" (img1 and img2
         are very different scenes); tells us how far the method generalizes
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import detect, evaluate, features, pipeline

OUT = Path("outputs"); FIG = Path("figures")
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
lines_out = []


def report(result, name):
    r = result
    line = (f"{name:>28} | count {r['n_pred']:>5} vs {r['n_true']:>5} "
            f"({r['pct_count_error']:+6.1f}%) | P {r['precision']:.2f}  "
            f"R {r['recall']:.2f}  F1 {r['f1']:.2f}")
    print(line); lines_out.append(line)


def baseline_dots(crop, radius, spacing, threshold):
    """Single bright-blob LoG kernel at one radius (no learning)."""
    score = detect.log_score_map(crop, radius)
    return detect.peaks_to_dots(score, 0.6 * spacing, threshold)


def tune_baseline(labeled, masks):
    """Pick the LoG threshold that maximizes mean F1 on the TRAINING regions.
    Radius and spacing come from the training barnacles (median over all)."""
    radius = float(np.median([L["radius"] for L in labeled]))
    spacing = float(np.median([L["spacing"] for L in labeled]))
    best = (-1, None)
    for t in np.linspace(0.0, 1.5, 31):
        f1s = []
        for L, m in zip(labeled, masks):
            d = pipeline.dots_inside(baseline_dots(L["crop"], radius, spacing, t), m)
            f1s.append(evaluate.match_dots(d, pipeline.dots_inside(L["gt"], m), L["radius"])["f1"])
        if np.mean(f1s) > best[0]:
            best = (float(np.mean(f1s)), t)
    return radius, spacing, best[1]


def split_experiment():
    labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
    feats = [features.compute_features(L["crop"])[0] for L in labeled]
    for tr, te in [("L", "R"), ("R", "L")]:
        train_m = [pipeline.half_mask(L["crop"].shape[:2], tr) for L in labeled]
        test_m = [pipeline.half_mask(L["crop"].shape[:2], te) for L in labeled]

        radius, spacing, t = tune_baseline(labeled, train_m)
        bundle = pipeline.train_model(labeled, valid=train_m, feats=feats)
        for L, f, m in zip(labeled, feats, test_m):
            gt = pipeline.dots_inside(L["gt"], m)
            d = pipeline.dots_inside(baseline_dots(L["crop"], radius, spacing, t), m)
            report(evaluate.match_dots(d, gt, L["radius"]), f"baseline {L['name']} {te}")
            dots, prob, _ = pipeline.detect_in_crop(bundle, L["crop"], feats=f, mask=m)
            dots = pipeline.dots_inside(dots, m)
            res = evaluate.match_dots(dots, gt, L["radius"])
            report(res, f"forest   {L['name']} {te}")
            if te == "R":  # save figures once
                x0 = L["crop"].shape[1] // 2 + 60
                zoom = (x0, x0 + 400, 500, 900)
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                evaluate.draw_overlay(L["crop"], dots, gt, res, ax=ax[0], zoom=zoom,
                                      title=f"{L['name']} held-out half: F1 {res['f1']:.2f}")
                ax[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
                ax[1].set_xlim(zoom[0], zoom[1]); ax[1].set_ylim(zoom[3], zoom[2])
                ax[1].set_title("barnacle probability"); ax[1].axis("off")
                plt.tight_layout(); plt.savefig(FIG / f"split_{L['name']}.png", dpi=70); plt.close()
                pg, tg = evaluate.tile_counts(dots, gt, L["crop"].shape[0], n_tiles=4)
                evaluate.plot_tile_errors(pg[:, 2:], tg[:, 2:], title=f"{L['name']} right half: count error per tile")
                plt.savefig(FIG / f"tiles_{L['name']}.png", dpi=70); plt.close()
        lines_out.append(f"  (fold train={tr}: baseline radius {radius:.1f}, threshold {t:.2f})")
        top = sorted(zip(bundle["clf"].feature_importances_, bundle["feature_names"]), reverse=True)[:6]
        lines_out.append("  top forest features: " + ", ".join(f"{n} {v:.2f}" for v, n in top))
        print(lines_out[-2]); print(lines_out[-1])


def cross_experiment():
    labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
    feats = [features.compute_features(L["crop"])[0] for L in labeled]
    for i, j in [(0, 1), (1, 0)]:
        bundle = pipeline.train_model([labeled[i]], feats=[feats[i]])
        dots, _, _ = pipeline.detect_in_crop(bundle, labeled[j]["crop"], feats=feats[j])
        report(evaluate.match_dots(dots, labeled[j]["gt"], labeled[j]["radius"]),
               f"forest {labeled[i]['name']} -> {labeled[j]['name']}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "split"
    {"split": split_experiment, "cross": cross_experiment}[which]()
    (OUT / f"metrics_{which}.txt").write_text("\n".join(lines_out) + "\n")
