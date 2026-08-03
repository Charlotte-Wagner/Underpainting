"""Image math for Underpainting.

Kept out of app.py on purpose: app.py runs Streamlit calls at import time, so
importing it from a plain terminal script is awkward. Everything here is pure
numpy in and numpy out, which means it can be run on a 4x4 array by hand.
"""

import cv2
import numpy as np

# Clustering settings. Hardcoded per cut list item 1: no sliders.
PALETTE_SIZE = 6
PALETTE_SAMPLE_PX = 200
KMEANS_SEED = 0


def posterize(gray, n_levels):
    """Map an 8-bit grayscale array onto n_levels evenly spaced values.

    Level 0 is always 0 (true black) and level n_levels - 1 is always 255
    (true white), so the study always spans the full range. That is the whole
    point: a value study that tops out at 170 is a muddy picture, not a study.

    The step between levels is 255 / (n_levels - 1), not 255 / n_levels.
    There are n_levels values but only n_levels - 1 gaps between them, and the
    step is a gap. Using n_levels here is the off-by-one that produces a
    plausible-looking image with the wrong lightest value.

    Args:
        gray: 2D uint8 array, values 0-255.
        n_levels: number of output values, at least 2.

    Returns:
        2D uint8 array of the same shape, containing at most n_levels
        distinct values drawn from an evenly spaced set anchored at 0 and 255.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2, got %r" % (n_levels,))

    step = 255.0 / (n_levels - 1)

    # Quantize: which of the n_levels levels is each pixel nearest to?
    # Result is an integer 0 .. n_levels - 1.
    levels = np.round(gray.astype(np.float32) / step)

    # Expand: put those level indices back on the 0-255 scale.
    # Level n_levels - 1 lands on exactly 255 by construction, not by clipping.
    out = levels * step

    # Clip is a float-safety net, not the thing making white white.
    # Round before the cast: astype truncates, so 127.5 would become 127.
    return np.round(np.clip(out, 0, 255)).astype(np.uint8)


def posterize_output_values(n_levels):
    """The exact uint8 values posterize can emit for a given n_levels.

    Useful for asserting against without needing an image that happens to
    contain every band.
    """
    step = 255.0 / (n_levels - 1)
    return np.round(np.arange(n_levels) * step).astype(np.uint8)


# --------------------------------------------------------------------------
# Color space
#
# There is exactly one sRGB -> Lab path in this project and this is it.
#
# cv2.cvtColor returns a different Lab scale depending on the dtype it is
# handed. Feed it uint8 and you get L scaled to 0-255 with a +128 offset on a
# and b. Feed it float32 in 0-1 and you get true CIE Lab, L 0-100 and a/b
# roughly +-127. Both run. Mixing them gives wrong distances and
# plausible-but-wrong paint matches, which is the S7 trap.
#
# The paint reference values are manufacturer-measured true CIE Lab, so the
# float32 path is not a preference, it is forced by the data. Everything that
# gets compared to a paint value comes through here.
# --------------------------------------------------------------------------


def srgb_to_lab(rgb_u8):
    """uint8 sRGB image (H, W, 3) -> float32 true CIE Lab (H, W, 3)."""
    arr = np.asarray(rgb_u8, dtype=np.uint8)
    return cv2.cvtColor(arr.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_srgb(lab):
    """float32 true CIE Lab (H, W, 3) -> uint8 sRGB (H, W, 3).

    Display only. Never feed the result of this back into a paint match; the
    round trip is lossy and reintroduces the mixed-scale problem by hand.
    """
    rgb01 = cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)
    # Round before the cast, same reason as in posterize: astype truncates.
    return np.round(np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample(rgb, max_px):
    """Shrink so the long edge is at most max_px. Never upscales.

    INTER_AREA, not INTER_LINEAR: when shrinking, INTER_AREA averages the
    pixels it is collapsing, which is what keeps the color statistics honest.
    INTER_LINEAR samples and aliases, which would bias the very histogram
    k-means is about to cluster.
    """
    height, width = rgb.shape[:2]
    scale = max_px / float(max(height, width))
    if scale >= 1.0:
        return rgb
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def extract_palette(rgb, k=PALETTE_SIZE, sample_px=PALETTE_SAMPLE_PX, seed=KMEANS_SEED):
    """Cluster an image's colors and return k swatches, most-used first.

    Three decisions live here:

    1. Cluster in Lab, not RGB, so the clusters correspond to colors a painter
       would treat as one mixture. Equal distances in Lab look equally
       different; equal distances in RGB do not.
    2. Sort by cluster population, not lightness or hue. A painter wants to
       know how much of the canvas each color covers, because that is what
       tells them how much to mix. Sorting by lightness makes a prettier strip
       and a less useful one.
    3. Seed the RNG. k-means picks its starting centroids randomly, so the
       same photo run twice returns different swatches. Measured drift on an
       unseeded run reached 41 Lab units, which is a different color, not a
       rounding wobble. Someone re-uploads their photo, gets a different
       palette, and concludes the app is broken, and they are right.

    Args:
        rgb: uint8 sRGB array (H, W, 3), any size.
        k: number of swatches.
        sample_px: long edge to downsample to before clustering.
        seed: RNG seed. Same seed and same image always gives the same result.

    Returns:
        List of k dicts, ordered by share descending, each with:
            "lab"   tuple of 3 floats, true CIE Lab. Use this for paint matching.
            "rgb"   tuple of 3 ints 0-255, for display.
            "hex"   "#rrggbb" string.
            "share" float 0-1, fraction of sampled pixels in this cluster.
    """
    small = downsample(np.asarray(rgb, dtype=np.uint8), sample_px)
    samples = srgb_to_lab(small).reshape(-1, 3)

    if samples.shape[0] < k:
        raise ValueError(
            "need at least %d pixels to find %d clusters, got %d"
            % (k, k, samples.shape[0])
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # Seed immediately before the call. cv2 seeds through global RNG state, so
    # the reset has to be adjacent to the thing it is protecting.
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k)

    # kind="stable" matters: argsort defaults to an unstable sort, so two
    # clusters with identical pixel counts could swap places between runs even
    # with the RNG seeded. Same class of bug, one layer down.
    order = np.argsort(-counts, kind="stable")

    total = float(labels.size)
    swatches = []
    for index in order:
        lab = centers[index]
        rgb_values = lab_to_srgb(lab.reshape(1, 1, 3)).reshape(3)
        swatches.append(
            {
                "lab": tuple(float(v) for v in lab),
                "rgb": tuple(int(v) for v in rgb_values),
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb_values),
                "share": float(counts[index]) / total,
            }
        )
    return swatches
