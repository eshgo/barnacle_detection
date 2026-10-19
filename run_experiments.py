"""
run_experiments.py: measure how well the barnacle counter works.

Usage:
    python run_experiments.py split    # main result: train on one half of both
                                       # labeled images, test on the other half
    python run_experiments.py cross    # train on one image, test on the other
    python run_experiments.py sweep    # dot spacing vs accuracy (uses split cache)
    add --fresh to ignore cached results and recompute

The experiments themselves live in src/experiments.py (shared with the
notebook). Results are cached in outputs/cache/ and summarized in
outputs/metrics_<experiment>.txt.

Why two main experiments: they answer different questions.
  split: "new photo that looks like the training photos" (the unseen photos
         resemble img1, so this is the relevant estimate)
  cross: "new photo that looks NOTHING like any training photo"
"""

import sys
from pathlib import Path

import pandas as pd

from src import experiments

OUT = Path("outputs")


def main(args):
    which = next((a for a in args if not a.startswith("--")), "split")
    use_cache = "--fresh" not in args
    if which == "split":
        res = experiments.split_experiment(use_cache)
        table = pd.DataFrame(res["rows"])
        extra = [f"fold {k}: baseline radius {f['baseline']['radius']:.1f}, "
                 f"threshold {f['baseline']['threshold']:.2f}" for k, f in res["folds"].items()]
    elif which == "cross":
        table = pd.DataFrame(experiments.cross_experiment(use_cache)["rows"]); extra = []
    elif which == "sweep":
        table = pd.DataFrame(experiments.spacing_sweep(use_cache=use_cache)); extra = []
    else:
        raise SystemExit(f"unknown experiment {which!r}; use split, cross or sweep")
    text = table.to_string(index=False, float_format=lambda v: f"{v:.2f}") + "\n" + "\n".join(extra)
    print(text)
    OUT.mkdir(exist_ok=True)
    (OUT / f"metrics_{which}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
