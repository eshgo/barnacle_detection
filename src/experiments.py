"""
experiments.py: the evaluation experiments, as functions that return results.

Used by both run_experiments.py (prints tables) and the notebook (plots
figures), so every reported number comes from one implementation.

Training forests is the slow part (a few minutes on one CPU core), so results
are cached in outputs/cache/. Delete that folder, or pass use_cache=False,
to recompute from scratch.
"""

from pathlib import Path

import joblib
import numpy as np

from . import data, detect, evaluate, features, pipeline

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "cache"


def cached(name, fn, use_cache=True):
    """Run fn() once and save its result; later calls load the saved copy."""
    path = CACHE / f"{name}.joblib"
    if use_cache and path.exists():
        return joblib.load(path)
    result = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path, compress=3)
    return result


def metrics_row(result, **labels):
    """Flatten a match_dots result into one table row (drops index arrays)."""
    keep = ["n_pred", "n_true", "pct_count_error", "precision", "recall", "f1"]
    return {**labels, **{k: result[k] for k in keep}}


# ---------------------------------------------------------------------------
# Baseline: one LoG kernel, threshold tuned on training regions
# ---------------------------------------------------------------------------

def baseline_dots(crop, radius, spacing, threshold):
    """Single bright-blob LoG kernel at one radius (no learning)."""
    score = detect.log_score_map(crop, radius)
    return detect.peaks_to_dots(score, 0.6 * spacing, threshold)


def tune_baseline(labeled, masks):
    """Threshold that maximizes mean F1 on the TRAINING regions. Radius and
    spacing are medians over the training images' barnacles."""
    radius = float(np.median([L["radius"] for L in labeled]))
    spacing = float(np.median([L["spacing"] for L in labeled]))
    best = (-1.0, None)
    for t in np.linspace(0.0, 1.5, 31):
        f1s = []
        for L, m in zip(labeled, masks):
            d = pipeline.dots_inside(baseline_dots(L["crop"], radius, spacing, t), m)
            f1s.append(evaluate.match_dots(d, pipeline.dots_inside(L["gt"], m), L["radius"])["f1"])
        if np.mean(f1s) > best[0]:
            best = (float(np.mean(f1s)), float(t))
    return radius, spacing, best[1]


# ---------------------------------------------------------------------------
# Experiment B (main): spatial split
# ---------------------------------------------------------------------------

def _split():
    labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
    feats = [features.compute_features(L["crop"])[0] for L in labeled]
    out = {"rows": [], "folds": {}}
    for tr, te in [("L", "R"), ("R", "L")]:
        train_m = [pipeline.half_mask(L["crop"].shape[:2], tr) for L in labeled]
        test_m = [pipeline.half_mask(L["crop"].shape[:2], te) for L in labeled]
        radius, spacing, t = tune_baseline(labeled, train_m)
        bundle = pipeline.train_model(labeled, valid=train_m, feats=feats)
        fold = {"baseline": {"radius": radius, "spacing": spacing, "threshold": t},
                "importances": dict(zip(bundle["feature_names"], bundle["clf"].feature_importances_)),
                "min_distance": bundle["min_distance"], "min_blob_area": bundle["min_blob_area"],
                "images": {}}
        for L, f, m in zip(labeled, feats, test_m):
            gt = pipeline.dots_inside(L["gt"], m)
            bd = pipeline.dots_inside(baseline_dots(L["crop"], radius, spacing, t), m)
            bres = evaluate.match_dots(bd, gt, L["radius"])
            dots, prob, _ = pipeline.detect_in_crop(bundle, L["crop"], feats=f, mask=m)
            dots = pipeline.dots_inside(dots, m)
            fres = evaluate.match_dots(dots, gt, L["radius"])
            out["rows"].append(metrics_row(bres, image=L["name"], test_half=te, method="baseline"))
            out["rows"].append(metrics_row(fres, image=L["name"], test_half=te, method="forest"))
            fold["images"][L["name"]] = {"gt": gt, "dots": dots, "result": fres,
                                         "baseline_dots": bd, "baseline_result": bres,
                                         "prob": prob.astype(np.float16), "radius": L["radius"]}
        out["folds"][f"train_{tr}"] = fold
    return out


def split_experiment(use_cache=True):
    return cached("split", _split, use_cache)


# ---------------------------------------------------------------------------
# Experiment A: train on one image, test on the other
# ---------------------------------------------------------------------------

def _cross():
    labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
    feats = [features.compute_features(L["crop"])[0] for L in labeled]
    out = {"rows": [], "probs": {}}
    for i, j in [(0, 1), (1, 0)]:
        bundle = pipeline.train_model([labeled[i]], feats=[feats[i]])
        dots, prob, _ = pipeline.detect_in_crop(bundle, labeled[j]["crop"], feats=feats[j])
        res = evaluate.match_dots(dots, labeled[j]["gt"], labeled[j]["radius"])
        out["rows"].append(metrics_row(res, trained_on=labeled[i]["name"], tested_on=labeled[j]["name"]))
        out["probs"][f"{labeled[i]['name']}->{labeled[j]['name']}"] = prob.astype(np.float16)
    return out


def cross_experiment(use_cache=True):
    return cached("cross", _cross, use_cache)


# ---------------------------------------------------------------------------
# Spacing sweep: does a bigger min_distance fix img2's overcount?
# ---------------------------------------------------------------------------

def spacing_sweep(distances=(6, 8, 10.5, 14, 18, 22, 26), use_cache=True):
    """Re-extract dots from the fold-L probability maps at several spacings.
    Cheap: reuses cached probabilities, no retraining."""
    def run():
        split = split_experiment()
        fold = split["folds"]["train_L"]
        rows = []
        for name, im in fold["images"].items():
            prob = im["prob"].astype(np.float32)
            m = prob > 0  # predicted region = the test half
            for md in distances:
                d, _ = detect.probability_to_dots(prob, 0.5, min_distance=md,
                                                  min_blob_area=fold["min_blob_area"])
                r = evaluate.match_dots(pipeline.dots_inside(d, m), im["gt"], im["radius"])
                rows.append(metrics_row(r, image=name, min_distance=md))
        return rows
    return cached("spacing_sweep", run, use_cache)


# ---------------------------------------------------------------------------
# Final model on the unseen photos
# ---------------------------------------------------------------------------

def final_counts(names=("unseen_img1.png", "unseen_img2.png"), use_cache=True):
    """Train on all of img1 + img2, then count each unseen photo."""
    def run():
        labeled = [pipeline.load_labeled(1), pipeline.load_labeled(2)]
        bundle = pipeline.train_model(labeled)
        out = {}
        for n in names:
            r = pipeline.count_image(bundle, data.load_image(n))
            out[Path(n).stem] = {"count": r["count"], "dots": r["dots"], "dots_raw": r["dots_raw"],
                                 "prob": r["prob"].astype(np.float16),
                                 "method": r["frame"]["method"], "corners": r["frame"]["corners"]}
        return out
    return cached("final_counts", run, use_cache)
