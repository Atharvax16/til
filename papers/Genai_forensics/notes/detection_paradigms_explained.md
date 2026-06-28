# `detection_paradigms.ipynb` — explained, cell by cell

A plain-English walkthrough of the companion notebook. Read this with the notebook open
side by side. For each cell you get: **what it does**, **the code explained**, and **what
the output means**. No prior signal-processing background assumed.

The notebook has **20 cells** top to bottom. They come in pairs: a *markdown* cell that
introduces an idea, then a *code* cell that demonstrates it. Below they're grouped by the
six paradigms.

---

## The one mental model to hold

An AI image and a real photo can look identical to your eye, but they differ in
*statistics* — patterns in the numbers behind the pixels. Each paradigm is a different
**lens** for making that hidden difference visible:

- pixels themselves (spatial),
- the image's "frequencies" (frequency),
- its leftover noise (fingerprint),
- region by region (patch),
- how well a model can rebuild it (training-free),
- what a smart vision model *notices* (multimodal).

Everything below is one of those six lenses.

---

## Cell 1 — Title (markdown)

Just the intro table listing the six paradigms and which part of your real pipeline
(`Etsy_final.ipynb`) each one maps to. Nothing to run. The key warning here: when the
notebook has no real data, it **simulates** fake images so you can still see every signal.
Those simulated fakes teach the *mechanism*, they are not real diffusion output.

---

## Section 0 — Setup & data

### Cell 2 — "## 0 — Setup & data" (markdown)
Tells you only basic libraries are needed; `cv2` and `anthropic` are optional.

### Cell 3 — imports & configuration (code)

```python
%matplotlib inline          # show plots directly inside the notebook
import io
from pathlib import Path
import numpy as np           # arrays / math
import matplotlib.pyplot as plt   # plotting
from PIL import Image        # load & convert images
from scipy import fftpack, ndimage   # FFT (frequencies) + image filters

try:
    import cv2               # optional; only used if present
except Exception:
    cv2 = None

REAL_DIR = None    # <-- put a folder path here to use YOUR real photos
FAKE_DIR = None    # <-- put a folder path here to use YOUR AI images
IMG_SIZE = 256     # every image is resized to 256x256 so shapes line up
RNG = np.random.default_rng(0)   # random generator with a fixed seed (reproducible)
```

**What it does:** loads tools and sets two switches. As long as `REAL_DIR`/`FAKE_DIR`
stay `None`, the notebook runs on built-in sample photos. Put real folder paths there and
it uses your data instead — **nothing else changes**.

**Output:** prints whether `cv2` is available. (It's optional, so "absent" is fine.)

### Cell 4 — helper functions & building the dataset (code)

This cell defines four small functions, then builds the `reals` and `fakes` lists.

```python
def _load_dir(d, limit=8):
    # collect up to 8 image files from a folder, resize each to 256x256,
    # return them as a list of numpy arrays (height x width x 3 colour channels)
```
Used only if you set `REAL_DIR`/`FAKE_DIR`.

```python
def _sample_reals():
    from skimage import data
    raw = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket()]
    # these are REAL photographs shipped inside scikit-image (no download)
    # resize each to 256x256 and return
```
The fallback "real" images.

```python
def simulate_fake(img, factor=3, amp=6.0, period=4.0):
    # 1. shrink the image by 3x then blow it back up with "nearest" sampling
    #    -> this leaves a blocky GRID, exactly what a generator's up-sampling does
    # 2. add a faint repeating wave pattern (sin) on top
    #    -> mimics the periodic "fingerprint" GANs leave
    # returns a fake-looking version of the real image
```
**This is the heart of the "no data needed" trick.** A real generator's giveaway is its
up-sampling step; we reproduce that giveaway cheaply so the later cells have something to
detect. It is a *teaching stand-in*, clearly not real AI output.

```python
def to_gray(img):
    # convert a colour image to a single-channel greyscale array of numbers
    # (most signal-processing steps work on one channel)
```

```python
if REAL_DIR and FAKE_DIR:
    reals, fakes = _load_dir(REAL_DIR), _load_dir(FAKE_DIR)   # your data
else:
    reals = _sample_reals()
    fakes = [simulate_fake(im) for im in reals]               # simulated
```

**Output:** `Loaded 4 real / 4 fake (source: ...)`. After this cell, `reals` and `fakes`
are lists of images every other section reuses.

### Cell 5 — show the images (code)

Draws a 2-row grid: real photos on top, their fake counterparts below. **Look:** the fakes
are slightly blurrier/blockier — that softness *is* the artifact later cells will exploit.

---

## Section 1 — Spatial domain (work directly on pixels)

### Cell 6 — intro (markdown)
The classic detector: show a neural net the raw pixels and let it learn "real vs fake."
Your **EfficientNet** branch is the real version. Here we use a fast stand-in.

### Cell 7 — spatial-statistics classifier (code)

```python
def patch_feats(img, n=300, size=24):
    # cut 300 small 24x24 squares from random positions in the image.
    # for each square, compute simple SPATIAL statistics per colour channel:
    #   - mean      = average brightness
    #   - std       = local contrast (how much pixels vary)
    #   - gradx/grady = average edge strength = SHARPNESS
    # returns one row of 12 numbers (4 stats x 3 channels) per square
```

`np.diff` is just "neighbouring pixel difference" — big differences mean sharp edges,
small differences mean a smooth/blurry area.

```python
X, y = [], []
for im in reals: X.append(patch_feats(im)); y.append(0...)   # label 0 = real
for im in fakes: X.append(patch_feats(im)); y.append(1...)   # label 1 = fake
# stack everything, split 70/30 into train/test
clf = RandomForestClassifier(...).fit(Xtr, ytr)   # learn the boundary
print(accuracy on the held-out test patches)
print(which features mattered most)
```

A **RandomForest** is a "committee of decision trees" — a simple, fast classifier. We feed
it the 12 statistics and it learns to tell real squares from fake squares.

**Output:** `validation accuracy: 0.91` and `most useful features: grady_R, grady_G, ...`.
**Meaning:** ~91% correct, and the most useful clues are the **gradient (sharpness)**
features — because the fakes are smoother. Lesson: *the pixel domain alone carries a
learnable signal.* (A real CNN discovers far richer versions of these features by itself.)

---

## Section 2 — Frequency domain (the image's "frequencies")

### Cell 8 — intro (markdown)
Any image can be rewritten as a sum of wave patterns — smooth areas are "low frequency,"
fine detail and edges are "high frequency." The tool for this is the **Fourier transform
(FFT)**. Generators leave tell-tale marks in this frequency view. **The right-hand plot is
literally the FFT feature in your pipeline.**

### Cell 9 — FFT and radial power spectrum (code)

```python
def radial_psd(gray):
    F = |FFT(gray)|**2            # how much energy at each frequency (a 2D map)
    # the centre of this map = low frequencies, the edges = high frequencies.
    # average the energy in rings at each distance from the centre
    # -> turns the 2D map into ONE curve: "energy vs frequency"
```

This "radial power spectrum" is a compact fingerprint of an image's frequency content —
the exact idea behind your FFT feature.

```python
Fr = log(|FFT(real)|)   # frequency picture of a real image
Ff = log(|FFT(fake)|)   # frequency picture of a fake image
# left panel: real spectrum, middle panel: fake spectrum
# right panel: averaged radial curve for real vs fake
```

**Output:** three panels. **Look:**
- Left vs middle: the **fake's frequency picture has a bright regular grid of dots** — the
  periodic artifact. The real one looks smooth.
- Right: the **fake curve sits higher at high frequencies / has bumps**. That gap is what
  a frequency-based detector keys on.

---

## Section 3 — Fingerprint based (leftover noise)

### Cell 10 — intro (markdown)
Like a camera leaves a sensor pattern, a generator leaves a faint, *consistent* noise
signature. Trick to reveal it: take the **noise residual** of many images and **average**
them. Real scene content cancels out to grey; the generator's repeated pattern survives.

### Cell 11 — residual fingerprint (code)

```python
def residual(img, sigma=1.0):
    g = to_gray(img)
    return g - gaussian_blur(g)     # original minus a blurred copy = the fine "noise"
```
`gaussian_filter` blurs the image; subtracting the blur leaves only the fine, high-detail
"noise" layer.

```python
fp_real = average of residual(image) over all REAL images
fp_fake = average of residual(image) over all FAKE images
# then show each fingerprint AND its FFT (frequency view)
```

**Output:** a 2x2 grid. **Look:** the **fake fingerprint's FFT shows bright structured
dots** (the repeated pattern survived averaging); the real one is featureless. That
structure is the generator's "signature." (Your ELA feature is a rough cousin of this.)

---

## Section 4 — Patch based (where is the fake-ness?)

### Cell 12 — intro (markdown)
Instead of one score for the whole image, score **each region** and draw a heatmap. This
matters for partially-edited images — e.g. a seller AI-retouches *half* a real photo. We
stitch **left = real, right = fake** and expect the heatmap to light up the right half.

### Cell 13 — spliced image + heatmap (code)

```python
spliced = reals[0].copy()
spliced[:, 128:] = fakes[0][:, 128:]   # replace the right half with the fake's right half
```

```python
def hf_energy(gray):
    # fraction of the patch's energy that sits at HIGH frequencies
    # (a quick "how much fine structure / artifact is here" score)
```

```python
P = 32
# slide a 32x32 window across the image; score every tile with hf_energy;
# store scores in a small grid 'heat'; then blow it up to image size and overlay
```

**Output:** two panels — the spliced image and its heatmap. **Look:** the heatmap is
**brighter over the fake (right) half**. That's the paradigm: the signal is *local*, and
scoring regions localizes the manipulation.

---

## Section 5 — Training-free (reconstruction error)

### Cell 14 — intro (markdown)
No labels, no training a detector. **DIRE** and **AEROBLADE** push an image through a
model that tries to rebuild it; real and generated images rebuild with *different* error.
We demonstrate the **mechanism** with a tiny "autoencoder" made of PCA.

### Cell 15 — PCA reconstruction error (code)

```python
def patch_matrix(imgs, size=16, stride=16):
    # chop every image into a tidy grid of 16x16 squares, flattened to rows of numbers
```

```python
pca = PCA(n_components=20).fit(real_patches)
```
**PCA** finds the 20 most common "ingredients" of *real* image patches. Think of it as a
compressor that only knows how to rebuild things it has seen — and it only saw real
texture.

```python
def recon_err(M):
    # compress each patch to 20 numbers then decompress; measure how wrong the rebuild is
    return mean_squared_error(original, rebuilt)

er = recon_err(real_patches)   # should be LOW (PCA knows real texture)
ef = recon_err(fake_patches)   # should be HIGHER (fake texture is unfamiliar)
# plot the two error distributions as histograms
```

**Output:** two overlapping histograms. **Look:** the **real and fake error distributions
sit apart** — error alone separates the classes, *with no labels used*. That's the whole
idea behind reconstruction-based detection.

> Sign caveat (in the markdown): our PCA learned *real* texture, so *real* rebuilds better.
> AEROBLADE uses the generator's own decoder, so *fake* rebuilds better. Opposite sign,
> identical mechanism.

### Cell 16 — DIRE pseudocode (markdown)
Shows, in a few lines, what the *real* methods do (run a diffusion model backwards then
forwards, compare). Nothing to run — it's the bridge from our toy PCA to the real
technique.

### Cell 17 — optional real AEROBLADE (code)
A commented-out block that, on a GPU with `diffusers` installed, would run the genuine
Stable-Diffusion-VAE version. It just prints a note by default — left for you to enable
later. Safe to skip.

---

## Section 6 — Multimodal reasoning (a model that explains)

### Cell 18 — intro (markdown)
Every method so far outputs a number. A **vision-language model (VLM)** can output a
verdict **plus reasons** — "the hand has six fingers," "background text is garbled." It
catches *semantic* mistakes that frequency math misses. The cell calls Claude's vision
model and asks for a **structured JSON** answer so you could plug it into your ensemble.

### Cell 19 — VLM scaffold (code)

```python
def encode_png(img):
    # turn an image into base64 text so it can be sent in an API request
```

```python
FORENSIC_PROMPT = "...act as an image-forensics expert, look for AI giveaways..."
SCHEMA = { verdict: authentic|ai_generated, confidence: number,
           evidence: [strings], reasoning: string }
```
The `SCHEMA` forces the model to answer in a fixed, machine-readable shape.

```python
def vlm_verdict(img, model="claude-opus-4-8"):
    if anthropic not installed: print a note, return None
    if no API key set:         print a note, return None
    otherwise: send the image + prompt to Claude, get back the JSON verdict
```

**Output:** by default `anthropic SDK not installed -> pip install anthropic`. That's
expected — the cell is built to **no-op safely** so the whole notebook still runs without
credentials. To make it live:

```
pip install anthropic
set ANTHROPIC_API_KEY in your environment, then re-run the cell
```

It will then print a structured verdict like:
```json
{ "verdict": "ai_generated", "confidence": 0.88,
  "evidence": ["over-smooth texture", "warped edge"], "reasoning": "..." }
```

---

## Cell 20 — Mapping back to your pipeline (markdown)

A summary table tying each paradigm to your real model and a "try next" idea. The two
*new* directions for your project are **Section 5** (add reconstruction error as an
ensemble member) and **Section 6** (a VLM evidence-vote + using the listing text).

---

## How to run it yourself

1. Open `notebooks/detection_paradigms.ipynb`.
2. Run cells **top to bottom** (each section reuses variables from Section 0).
3. To use your own images, set `REAL_DIR` and `FAKE_DIR` in Cell 3 to two folders, then
   re-run — every plot becomes the real story instead of the simulated one.
4. Section 6 stays inert until you install `anthropic` and set `ANTHROPIC_API_KEY`.

## One-line summary of each section

1. **Spatial** — pixels are enough to learn from (fakes are smoother).
2. **Frequency** — fakes have a tell-tale grid in their Fourier picture (your FFT feature).
3. **Fingerprint** — averaging noise reveals the generator's repeated signature.
4. **Patch** — score regions to find *where* an image was faked.
5. **Training-free** — measure how well a model can rebuild it; no labels needed.
6. **VLM** — ask a smart model *why* it's fake, in plain language.
