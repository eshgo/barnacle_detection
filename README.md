# Barnacle Counter (DALI Data Challenge)

Counts barnacles inside the green frame of a field photo, and produces an
overlay a scientist can check and correct instead of counting by hand.

**Approach:** find the frame's biggest rectangle → describe every pixel with
kernel filters (blur, edges, blob detectors, color) → a random forest predicts
"is this pixel a barnacle?" → turn the probability map into one dot per barnacle.

**Results (held-out half of each labeled image):** on img1-like scenes, count
within 2.5–3.3% (F1 0.78–0.84). On img2's very different dark scene, a ~3×
overcount (F1 ≈ 0.44). Full story, including everything that failed, in `LOG.md`.

**Unseen photos:** unseen_img1 → 1482 barnacles; unseen_img2 → 194.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the challenge images (img1.png, mask1.png, ...) in `data/`.

## Run
```bash
python count_barnacles.py                # count unseen_img1 + unseen_img2 → outputs/
python count_barnacles.py my_photo.png   # count any photo
python run_experiments.py split          # main evaluation (≈2–3 min on 1 CPU)
python run_experiments.py cross          # train on one image, test on the other
```

## Files
| File | Role |
|---|---|
| `src/data.py` | load images; turn hand-drawn mask outlines into one dot per barnacle |
| `src/frame.py` | find the teal frame, crop + straighten its biggest rectangle |
| `src/features.py` | kernel filter bank (the per-pixel features) |
| `src/detect.py` | baseline blob detector; random forest; probability map → dots |
| `src/evaluate.py` | one-to-one dot matching, metrics, overlay and tile plots |
| `src/pipeline.py` | the end-to-end steps shared by both scripts |
| `count_barnacles.py` | the program a scientist would run |
| `run_experiments.py` | reproduces every number in LOG.md |
| `LOG.md` | decisions, successes, failures, and what I'd do next |