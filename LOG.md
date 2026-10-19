# Lab Log: Counting Barnacles

This project was built iteratively with help from Claude, which I used to write and debug code and to talk through design decisions. However, all design decisions, overall strategy, and final review was done by me. Claude recorded and summarized large portions of our conversations and decision making in this file.

Every number below comes from code in this repo and can be reproduced with `run_experiments.py` and `count_barnacles.py`. The notebook (`notebooks/barnacle_counting.ipynb`) tells the same story with figures.

---

## Initial Learning and Idea Generation Resources

- [Cropping an Image Using OpenCV](https://opencv.org/cropping-an-image-using-opencv/)
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Kernel (image processing)](https://en.wikipedia.org/wiki/Kernel_(image_processing))

## Initial Image Observations

- Barnacles often have an eye-like shape on top.
- Some barnacles are sideways, so the opening on top is not clearly visible.
- Barnacles vary in size.
- The frames are both green, but their shades differ slightly. This should be manageable.
- Barnacles in the two photos differ in color.
- `unseen_img2` does not have a green frame. If the detector cannot find one to crop, it should leave the image uncropped.
- `unseen_img1` looks similar to `img1`.
- `unseen_img2` looks straightforward; there do not appear to be any ambiguous barnacles.

## The idea

Scientists photograph a wire frame on a rock and then count every barnacle
inside it by hand, often over 1,000 per photo. Full automation is unrealistic
with only two labeled photos, so the goal is a **first draft**: the system
proposes one dot per barnacle, and a person checks and corrects the dots
instead of counting from zero.

I broke the system into four subtasks, each with a clear input and output:

| Subtask | Input | Output | File |
|---|---|---|---|
| Ground truth | hand-drawn outline masks | one (x, y) dot per barnacle | `data.py` |
| Frame cropping | raw photo | straightened image of the frame's biggest rectangle | `frame.py` |
| Detection | cropped image | one dot per predicted barnacle | `features.py`, `detect.py` |
| Evaluation | predicted dots + true dots | count error, precision, recall, F1, plots | `evaluate.py` |

The core idea: if every barnacle becomes a dot, counting is trivial, and
evaluating is just comparing my dots with the scientists' dots.

---

## 1. Ground truth: turning outlines into dots

This took longer than anything else, and it was worth it. Every metric in
the project depends on it, so a bug here would make every number wrong.

### What's actually in the masks

* The masks are RGBA PNGs: **transparent white background**, lines drawn in
  blue. Since the background is white, color can't separate lines from
  background, but transparency can, so line pixels are those with alpha > 0.
  Alpha only takes two values, so the lines have crisp edges.
* Overlaying masks on photos showed that **each outline traces the
  barnacle's central plates (the operculum)**, not the whole white shell.
* **Empty shells (dead barnacles, just a dark hole) are not outlined.** So the
  real target is "live barnacles with visible plates," which makes the task
  harder: a classifier has to tell a live barnacle from an empty shell that
  looks very similar.

### Attempt 1: count the empty pockets between lines

My first version ignored the lines and counted the closed pockets of empty
space they enclose, one pocket per barnacle. I chose this because I assumed
touching outlines would be the main problem, and pockets stay separate even
when outlines touch.

**What went wrong:** I designed around an assumed problem, not the real one.
Looking at the overlay, the real problems were outlines with gaps (a gapped
outline encloses no pocket, so that barnacle silently vanished) and stray
floating pixels.

### Attempt 2: trace outlines with `findContours` and close gaps globally

I switched to tracing outlines directly, closing small gaps with a
morphological close (dilate, then erode), and repairing outlines with big
gaps using their convex hull (the "rubber band" shape around them).

**What went wrong:** I ran the gap closing on the whole image at once, so it
also bridged the narrow spaces *between* neighboring barnacles and fused
them. On a synthetic test of 70 tightly packed circles, it found **7 dots at
gap size 5 and 1 dot at gap size 9.**

### Attempt 3: close gaps one outline at a time

Each connected cluster of line pixels gets cut out onto its own canvas, so
closing can't reach a neighbor. The same test went back to **70 of 70 at
every gap size.**

### Attempt 4: handle touching outlines, bubbles, and specks

Then I noticed some outlines do touch, though never more than two at once. A
touching pair forms one cluster, but each closed outline still encloses its
own hole. So the rule became **count holes, not clusters.** Holes below a
minimum size are "bubbles," tiny gaps inside the drawn line itself, and don't
count.

I built a synthetic test with every case and its combinations: touching pairs,
an open outline touching a closed one, two open outlines touching, bubbles,
and specks. It found a real bug. When an open outline touched a closed one,
the closed barnacle's claimed area cut the open outline into two pieces,
which became two dots. The fix used the "at most two touch" rule: a cluster
with one real hole can only have one open outline attached, so all its
leftover pieces belong to one barnacle.

### Attempt 5: the real masks taught me three more things

Running on the real masks and zooming in revealed problems the synthetic
tests couldn't:

1. **Line thickness varies within one mask.** mask1's top-left region has
   strokes 8 to 10 px thick; elsewhere they're about 3 px. I had measured one
   width for the whole mask (3.6 px), so in the thick region only the inner
   part of each stroke was assigned to its barnacle. The leftover outer rim
   became slivers, which got hull-repaired into **fake barnacles sitting just
   below real ones.**
   *Fix:* measure line width for each cluster separately, and assign each
   stroke pixel to the nearest hole.
2. **Gap closing destroys small barnacles.** With closing on, the hole-size
   setting had *no effect at all*, because closing filled every hole narrower
   than about 5 px before the check ran, including the interiors of tiny
   barnacles. That's why 109 barnacles were labeled "repaired" when only 38
   clusters actually lacked a hole.
   *Fix:* find holes with and without closing, and keep whichever version
   finds more. Repaired barnacles in mask1 dropped from 109 to 14.
3. **Pen-overlap loops.** Tiny closed loops appear where the annotator's
   stroke overlapped itself at the start or end of an outline. They're bigger
   than bubbles, so the bubble filter misses them.
   *Fix:* drop any shape smaller than 15% of the median barnacle's area. I
   checked every shape this filter removed (7 in mask1, 22 in mask2) by eye,
   and all of them were junk.

How common each problem is in the real masks:

* **Bubbles are everywhere:** 358 of mask2's 397 outline clusters contain at
  least one; nearly 1,000 in total. Without the bubble filter, mask2's count
  would roughly double.
* **Touching pairs:** 15 in mask1, 1 in mask2. **No cluster has three or more
  holes,** which confirms "at most two touch" from the data itself.
* **Broken outlines:** I checked for pairs of repaired dots close together,
  which is what an outline broken into two pieces would produce. There are
  none.

**Final ground truth: 1,733 barnacles in mask1 (14 hull-repaired) and 398 in
mask2 (all closed).**

**Known limitation:** when two *open* outlines touch, the code splits them at
their narrowest point, but hull repair puts each dot 8 to 12 px off-center,
because a hull of an outline missing a third of its arc is lopsided. This
case never occurs in the real masks. I tried k-means for the split first; on
thin rings it sometimes split diagonally, because rings aren't the compact
blobs k-means assumes, so I switched to cutting at the narrowest point along
the long axis.

---

## 2. Frame: finding the biggest rectangle

The frame is a grid of rectangles, and only the biggest one is counted. The
biggest rectangle is the largest region of non-wire completely enclosed by
wire. So: make a "wire pixel" mask, find all enclosed holes with
`findContours`, take the largest, and simplify it to 4 corners. A perspective
warp then straightens it into a square. Photos aren't taken perfectly
straight down, so a simple bounding-box crop would include rock outside the
frame.

**First run: complete failure.** The "largest hole" on every photo was tiny,
6 to 79 px. My color range was a guess (hue 35 to 90, the usual "green"), but
the wire is **teal.** When I sampled the wire on all four sides of every
photo, its hue was 87 to 98, mostly outside my range. It's also very dark in
img1's shade (brightness as low as 31, below my minimum of 40), but strongly
saturated everywhere (125 to 200). After switching to the measured range
(hue 75 to 110, saturation ≥ 80, brightness ≥ 20), the wire mask came out
clean on every photo.

**How I verified it:**

* All 1,733 and all 398 ground-truth barnacles land inside the detected crops.
* The crop sizes (1460 px for img1) closely match the provided pre-cropped
  `masked_img` files (1449 px).
* Visual check: `figures/frame_detection.png`.

**No frame:** `unseen_img2` has no frame. If no enclosed region covers at
least 5% of the photo, the program counts the whole image. That's the
sensible default when there's nothing to restrict the count.

---

## 3. A detour about scale, and a lesson

`unseen_img2`'s barnacles looked about twice as big as everyone else's. A
model that learned "barnacles are about this many pixels wide" would struggle
on a zoomed-in photo. Framed photos have a built-in ruler (the frame), but
`unseen_img2` doesn't. So I tried to estimate barnacle size from the image
content alone:

* **Attempt 1: blob kernels at many sizes, pick the strongest.** It picked the
  smallest size every time, even though true barnacle radii differ almost 2×
  between img1 and img2. Fine rock texture and plate edges dominate.
* **Attempt 2: autocorrelation width** (how far you can shift the image
  before it stops resembling itself). The estimate changed by a factor of 6
  depending on one filter setting. It was measuring my filter, not the
  barnacles.

Then I finally asked the basic question: where did `unseen_img2` come from?
Template matching found it **inside `unseen_img1`, unscaled, with a perfect
match score of 1.000** at pixels (1500, 1000) to (2000, 1500). It only looked
zoomed because it was displayed larger. There was no scale problem at all.

**Lesson:** I spent two experiments solving a problem that didn't exist.
Check the data first. The silver lining: since `unseen_img2` is part of
`unseen_img1`, the two counts should roughly agree. With no labels for
the unseen photos, that's the only test of the final model available.

---

## 4. Detection

### Approach

Following how the tool ilastik classifies pixels:

1. **Kernels describe each pixel's neighborhood.** For each of 5 scales
   (σ = 1, 2, 4, 8, 16 px), the code computes a Gaussian blur ("how bright is
   it around here"), gradient magnitude ("is there an edge"), and a Laplacian
   of Gaussian ("is this the center of a blob this size"). Two Lab color
   channels are added, for 17 numbers per pixel.
2. **A random forest learns** from the masks which combinations of those
   numbers mean "barnacle pixel," and outputs a probability map.
3. **Probability map to dots:** threshold at 0.5, and remove tiny blobs. Then
   a distance transform turns each blob into a hill, and each hilltop becomes
   one dot.

I added σ = 16 after the first real-data run: img2's barnacles (radius about
19 px) were bigger than anything my largest kernel (σ = 8) responds to.
`log_16` then became one of the forest's most important features.

**Baseline for comparison:** a single blob kernel at one size, with its
threshold tuned on training data. If the forest can't beat this, the extra
complexity isn't earning its keep.

**Speed:** I measured before deciding whether to shrink images. On one CPU
core, features take 0.5 s and prediction about 6 s for a full crop. So
everything runs at full resolution, with no detail lost.

### Evaluation design

* **Matching:** each predicted dot is paired one-to-one with a true dot
  (Hungarian algorithm), within half a typical barnacle diameter. One-to-one
  matters: otherwise, one dot between two barnacles could "find" both.
* **Two kinds of metric:** count error, which is what the scientists record,
  and precision/recall/F1, which catch a lucky count where misses and false
  alarms cancel out.

### Experiment A: train on one image, test on the other. FAILED.

| Trained on → tested on | Count | F1 |
|---|---|---|
| img1 → img2 | 1662 vs 398 (+318%) | 0.25 |
| img2 → img1 | 503 vs 1733 (−71%) | 0.08 |

Each model's count landed near its *training* image's count. It learned
"how dense barnacles are in my photo," not "what a barnacle looks like."
The probability maps (`figures/cross_image_failure.png`) show it: trained on
img1, the forest lights up brown rock in img2; trained on img2, it produces a
vague haze over img1.

That makes sense. The photos are very different: img1 has pale barnacles
packed tightly on light rock, while img2 has sparse brown barnacles on wet,
dark red rock. With one example of each, nothing can learn what they have in
common. **More varied labeled photos are the single most valuable next step.**

### Experiment B: spatial split. The main result.

The unseen photos look much more like img1 than img2, so the real question is
how the model does on new photos similar to its training photos. So I trained
on the left half of *both* images and tested on the right halves, then
swapped. There's a 60 px buffer between halves: the largest kernel looks
about 48 px around each pixel, so without a gap, test pixels near the middle
would share context with training pixels.

| Held-out half | Method | Count (pred vs true) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| img1 right | **forest** | 835 vs 808 (**+3.3%**) | 0.83 | 0.86 | **0.84** |
| img1 left | **forest** | 787 vs 768 (**+2.5%**) | 0.77 | 0.79 | **0.78** |
| img1 right | baseline | 297 vs 808 (−63%) | 0.42 | 0.15 | 0.23 |
| img1 left | baseline | 331 vs 768 (−57%) | 0.39 | 0.17 | 0.24 |
| img2 right | forest | 658 vs 220 (+199%) | 0.29 | 0.85 | 0.43 |
| img2 left | forest | 461 vs 153 (+201%) | 0.29 | 0.88 | 0.44 |
| img2 right | baseline | 299 vs 220 (+36%) | 0.43 | 0.59 | 0.50 |
| img2 left | baseline | 361 vs 153 (+136%) | 0.28 | 0.66 | 0.39 |

**On img1-like scenes, the forest is within 2.5 to 3.3% of the true count
and finds about 80 to 86% of barnacles in the right place.** It beats the
single-kernel baseline by a wide margin. The baseline has to choose *one*
barnacle size, and the tuned value (radius 14.7) is a compromise between
img1's barnacles (10.5) and img2's (18.8) that fits neither.

**On img2, the forest overcounts about 3×.** It still finds 85 to 90% of the
real barnacles, but only 29% of its dots are correct. In one fold the plain
baseline actually beats it. Zooming in (`figures/split_img2.png`) shows why:
the probability is high across *whole shells*, including empty ones, not just
the operculum the scientists outlined.

### Trying to fix img2's overcount

* **Wider spacing between dots:** helps img2 (at spacing 22 px, +199% drops
  to +56%) but hurts img1 (loses about a third of its barnacles). No single
  value works for both.
* **Estimating barnacle size from the predicted blobs, per image:** fails.
  img2's predicted blobs are *smaller* than img1's (7.5 vs 12.4 px), even
  though its barnacles are nearly twice as big, because the blobs are
  fragments of shell rather than whole opercula.

So img2's problem isn't dot spacing. It's that the pixel classifier can't
tell "operculum" from "shell" in that scene. I kept the spacing derived from
training data (the smallest typical barnacle radius), which is right for
img1-like scenes.

### Attempted but not finished: a second-stage "candidate judge"

The idea: let stage 1 over-generate candidate dots, then train a second
forest to judge each *candidate* using information a single pixel can't see.
That includes probability averaged over several neighborhood sizes (a real
operculum is a small, sharp peak; a shell is broad) and how crowded it is by
other candidates. To keep stage 2 from trusting overconfident probabilities,
it has to train on stage-1 predictions from a model that never saw those
pixels.

I wrote it, but the nested experiment exceeded my compute time limit (one CPU
core, 5 minutes per run) and never finished. **I removed the code rather than
ship something untested.** It's the first thing I'd try next.

---

## 5. Final results on the unseen photos

The final model trains on **all** of img1 and img2 (about 30 s).

| Photo | Count | Frame |
|---|---|---|
| unseen_img1 | **1,482** | found (1546 × 1546 px crop) |
| unseen_img2 | **194** | none, so the whole image was counted |

**Consistency check:** the same 500 × 500 square inside unseen_img1 gets 175
dots, versus 194 when counted on its own, an **11% difference.** Some
difference is expected: barnacles cut by the square's edge get counted when
it stands alone, and kernels near an image edge see less context.

Looking at the detection images: nearly every live barnacle gets one dot on
its operculum, and empty shells and a snail are mostly skipped. The errors I
can see are a few large barnacles with two dots, some dots on shell rims or
empty shells, and misses along the image edges. That's consistent with the
held-out img1 results, which is reassuring, because these photos resemble img1.

**How much to trust 1,482:** with no labels for these photos, I can't measure
its accuracy directly. The img1 held-out results suggest a count error of a few
percent *if* unseen_img1 behaves like img1. The quickest way to verify would
be to hand-count one quarter of the crop and compare.

---

## 6. How the code is organized, and why

The five original modules each do one step: `data.py`, `frame.py`,
`features.py`, `detect.py`, `evaluate.py`. As the project grew, I added three
layers on top of them, each for a specific reason:

* **`src/pipeline.py` calls the steps in order.** Load and crop a labeled
  image, move its ground truth into the crop, train, turn probabilities into
  dots. Before this existed, that sequence was copied into each experiment
  script, and copies drift apart. Then the numbers in this log would describe
  a slightly different program from the one a scientist runs.
* **`src/experiments.py` holds the experiments as functions** that return
  results: the spatial split, the cross-image test, the dot-spacing sweep, and
  the final unseen counts. Both `run_experiments.py` (which prints tables)
  and the notebook (which draws figures) call these same functions, so the
  notebook can't quietly disagree with this log.
* **Two entry points:** `count_barnacles.py` is the tool a scientist would
  use (photos in, counts and review images out); `run_experiments.py`
  reproduces the evaluation (`split`, `cross`, or `sweep`).

**Caching.** Training the forests is the slow part: the split experiment takes
2 to 3 minutes on one CPU core. So `experiments.py` saves each experiment's
results to `outputs/cache/` the first time and loads them afterwards. The
cache is gitignored; delete it, or run `run_experiments.py split --fresh`, to
recompute from scratch. To keep it small, probability maps are stored at half
precision. That has one visible side effect: in the spacing sweep, img1's
count at the chosen spacing is 836 instead of 835, because one pixel landed
on the other side of the 0.5 threshold.

**Two practical lessons from this part:**

* **Refactoring needs a check.** After moving the experiments into
  `experiments.py`, I reran the split experiment and confirmed every number
  matched the earlier run exactly before trusting the new code.
* **Seeding makes results reproducible.** Every random step (pixel sampling,
  forest training) uses a fixed seed. Running `count_barnacles.py` from a
  clean copy of the repo, with no cached model, gives the same 1,482 and 194.

---

## 7. What I didn't do, and why

* **No deep learning (CNN, U-Net, Mask R-CNN).** With two labeled photos, a
  network would mostly memorize them. Experiment A shows even a forest can't
  bridge the gap between two scenes. A forest trains in seconds, runs on a
  laptop, and its feature importances show what it relies on.
* **No Segment Anything.** It would be an interesting comparison, but it can't
  run on this CPU-only setup in reasonable time.
* **No full-image counting by area** (total barnacle area ÷ typical size). On
  synthetic tests it overcounted, because predicted regions come out slightly
  fatter than labels.
* **No rule for barnacles cut by the frame edge.** Field ecologists often
  count edge barnacles on two sides only. I don't know this team's rule, so
  barnacles whose centers fall inside the crop count. This should be
  confirmed with the scientists.
* **No interactive review tool.** The detection overlays are the starting
  point for one (see below).

---

## 8. How this would fit the scientists' workflow

The output isn't meant to replace counting, but to change it from *counting*
1,500 barnacles to *checking* 1,500 dots, which is much faster. A review tool
would:

1. show the photo with dots;
2. let the scientist click to add a missed barnacle or remove a wrong dot;
3. highlight areas where the model is least sure, such as probability near
   0.5, or tiles where dots are unusually dense. `evaluate.tile_counts` and
   `plot_tile_errors` already compute per-tile errors;
4. save the corrected dots, which become **new training labels**. Point
   labels are far cheaper to make than traced outlines, and every corrected
   photo makes the next prediction better.

---

## 9. Is this worth pursuing?

**For img1-like scenes, yes.** A few percent count error with simple,
explainable tools on two labeled photos is a solid starting point, especially
as a first draft for human review.

**For very different scenes like img2, not yet.** The fixes I'd try, in order:

1. **More labeled photos covering different scenes.** This is the biggest
   lever, since the failures are all about scenes the model hasn't seen.
2. **The stage-2 candidate judge** described above.
3. **Separate labels for "live barnacle" and "empty shell,"** so the model
   learns the difference instead of guessing it.

---

## 10. What I learned

* **Look at the data before designing around it.** This was the most
  repeated lesson of the project: pockets versus outlines, the assumed green
  of a teal wire, the "scale problem" that wasn't. Every time, the fix came
  from zooming in on real pixels rather than thinking harder.
* **Small preprocessing steps have big hidden side effects.** Gap closing was
  meant to fix gaps; it also merged neighbors (when global) and erased small
  barnacles (when local). I only caught both by comparing settings and
  looking at the outputs.
* **An honest evaluation design matters more than the model.** Training on
  one image and testing on the other gave terrible numbers, and the spatial
  split gave good ones. Both are true; they answer different questions. The
  buffer between halves and the out-of-fold idea both came from thinking
  about what the model could secretly "see."
* **Kernels are surprisingly powerful features.** A random forest on 17
  hand-designed kernel responses got within a few percent on img1-like scenes,
  and its feature importances made sense: large-scale blob detectors and color
  mattered most.

---

## 11. Sources

* OpenCV-Python tutorials: color spaces (HSV), morphological
  transformations, contours and contour hierarchy (`RETR_CCOMP`),
  geometric transformations (perspective warp), template matching.
* scikit-image documentation: `peak_local_max`, blob detection with LoG.
* scikit-learn documentation: `RandomForestClassifier`, feature importances.
* SciPy documentation: `linear_sum_assignment` (Hungarian algorithm),
  `distance_transform_edt`, `cKDTree`.
* ilastik documentation on pixel classification: the idea of a random forest
  on a bank of filter responses.
* Lindeberg's work on scale-space and automatic scale selection: the theory
  behind the LoG scale-estimation attempt, and why its σ²-normalization exists.
* Claude, used throughout to write, test, and debug code and to
  talk through design decisions.
