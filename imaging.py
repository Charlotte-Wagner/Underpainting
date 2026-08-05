"""Image math for Underpainting.

Kept out of app.py on purpose: app.py runs Streamlit calls at import time, so
importing it from a plain terminal script is awkward. Everything here is pure
numpy in and numpy out, which means it can be run on a 4x4 array by hand.
"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps

# Clustering settings. Hardcoded per cut list item 1: no sliders.
PALETTE_SIZE = 6
PALETTE_SAMPLE_PX = 200
KMEANS_SEED = 0


def load_rgb(image_bytes, max_dimension):
    """Decode raw image bytes into an EXIF-corrected, resized, uint8 sRGB array.

    Shared by app.py's main display flow and the cached model-call path, so
    both start from the exact same pixels: same rotation, same resize, same
    array. Never upscales, same as .thumbnail()'s own behavior.

    HEIC/HEIF decoding depends on pillow_heif.register_heif_opener() having
    already run once, which app.py does at import time. This function does
    not call it itself: registration is global to Pillow, not scoped to
    whichever file happens to call Image.open.

    Args:
        image_bytes: raw bytes of an uploaded image file.
        max_dimension: long-edge cap in pixels.

    Returns:
        uint8 sRGB array (H, W, 3).
    """
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((max_dimension, max_dimension))
    return np.array(image)


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


def value_range(gray):
    """The actual darkest and lightest pixel in a grayscale array.

    Not assumed to be 0 and 255. A hazy or low-contrast photo genuinely
    doesn't reach either end, and that's information the rubric prompt
    needs, not a bug to correct for.

    Args:
        gray: 2D uint8 array.

    Returns:
        (min, max) as plain Python ints.
    """
    return int(gray.min()), int(gray.max())


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


def dominant_temperature(palette):
    """Share-weighted average Lab b value across a palette's swatches.

    The Lab b axis runs blue (negative) to yellow (positive), close enough
    to painting's warm/cool axis to be useful, and free: it reuses the same
    Lab values the palette already carries rather than a new color model.
    Ignores the a axis (green/red) entirely, so this is a simplification,
    not a full color-temperature measurement.

    Weighted by share rather than a plain mean over the k swatches, so a
    palette dominated by one large cool sky and a few small warm accents
    reports as cool, which is what the canvas actually reads as.

    Args:
        palette: list of dicts from extract_palette, each with "lab" and
            "share".

    Returns:
        Float. Positive leans warm (toward yellow), negative leans cool
        (toward blue).
    """
    return sum(swatch["lab"][2] * swatch["share"] for swatch in palette)


# --------------------------------------------------------------------------
# Stage generation
#
# Four independent views of the same source image, one per axis of
# information: value count, then color count, then detail, then nothing.
# Each is computed straight from the untouched rgb array, never from another
# stage's output. Chaining would compound each stage's own approximation
# into the next one, and the last stage would drift off the actual photo
# instead of being it.
# --------------------------------------------------------------------------

STAGE_VALUE_LEVELS = 3
STAGE_BLUR_FRACTION = 0.02

# Labels for build_stages's four outputs, in order. Single source of truth:
# app.py uses these as filmstrip captions and rubric.py uses the same
# strings as the step labels it asks the model for, so the written guide
# and the images point at each other instead of running two numbering
# schemes that can drift apart.
STAGE_CAPTIONS = ["1 · Values", "2 · Color masses", "3 · Soft focus", "4 · Full detail"]


def value_block_in(rgb, n_levels=STAGE_VALUE_LEVELS):
    """Stage 1: values only. Grayscale, posterized, put back on 3 channels.

    Reuses posterize at the same n_levels already validated in S5 as the one
    that reads as connected massing, so this is the same value study, not a
    new decision.
    """
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(posterize(gray, n_levels), cv2.COLOR_GRAY2RGB)


def recolor_to_palette(rgb, palette):
    """Stage 2: color count only. Every pixel snapped to its nearest palette color.

    Distance is Lab, the same space the palette itself was clustered in, so a
    pixel lands in the cluster it would have joined had it been part of the
    k-means input. Matched at full resolution, not the 200px sample the
    palette was clustered from: the centroids describe the sample, but every
    full-res pixel still has a well-defined nearest one.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: list of dicts from extract_palette (each needs "lab" and "rgb").

    Returns:
        uint8 sRGB array (H, W, 3), containing only the palette's own colors.
    """
    lab = srgb_to_lab(rgb)
    height, width = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    centers = np.array([p["lab"] for p in palette], dtype=np.float32)
    colors = np.array([p["rgb"] for p in palette], dtype=np.uint8)

    # One (N,) distance array per center, not one (N, k, 3) array of diffs:
    # k is small (6) but N is up to a few million pixels, and the (N, k, 3)
    # version peaks at k times the memory for no benefit.
    distances = np.stack(
        [np.linalg.norm(flat - center, axis=1) for center in centers], axis=1
    )
    nearest = np.argmin(distances, axis=1)
    return colors[nearest].reshape(height, width, 3)


def soften(rgb, blur_fraction=STAGE_BLUR_FRACTION):
    """Stage 3: detail only. Gaussian blur scaled to the image's own size.

    A fixed pixel radius means something different on a 400px image than a
    1200px one. Expressing the radius as a fraction of the long edge keeps
    the amount of softening visually comparable across photo sizes.
    """
    height, width = np.asarray(rgb).shape[:2]
    long_edge = max(height, width)
    radius = max(1, int(round(long_edge * blur_fraction)))
    kernel = radius * 2 + 1  # GaussianBlur requires an odd kernel size.
    return cv2.GaussianBlur(np.asarray(rgb, dtype=np.uint8), (kernel, kernel), 0)


def build_stages(rgb, palette):
    """The four stages, in build order: values, color masses, soft focus, full detail.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: the same palette extract_palette returned for this rgb, so
            stage 2 uses the identical clusters shown in the swatches panel.

    Returns:
        List of 4 uint8 sRGB arrays (H, W, 3), same size as rgb. The last one
        is rgb itself, unmodified, not a fourth approximation of it.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    return [
        value_block_in(rgb),
        recolor_to_palette(rgb, palette),
        soften(rgb),
        rgb,
    ]


def cross_fade_frames(stages, fade_frames):
    """Linear cross-fade between each consecutive pair of stages.

    Blended in uint8 RGB, not Lab: an animation frame is a display artifact
    on its way to a GIF palette, not a color measurement, and Lab would add a
    round trip through lab_to_srgb for every intermediate frame with nothing
    visible to show for it.

    Args:
        stages: list of uint8 RGB arrays, all the same shape.
        fade_frames: number of interpolated frames generated strictly
            between each pair (the endpoints are included once each, not
            repeated at both ends of their neighboring fades).

    Returns:
        Frames in playback order: stages[0], fade_frames frames toward
        stages[1], stages[1], fade_frames frames toward stages[2], and so on
        through stages[-1]. Length is len(stages) + (len(stages) - 1) * fade_frames.
    """
    frames = [stages[0]]
    for i in range(len(stages) - 1):
        start = stages[i].astype(np.float32)
        end = stages[i + 1].astype(np.float32)
        for step in range(1, fade_frames + 1):
            t = step / (fade_frames + 1)
            blended = start * (1.0 - t) + end * t
            frames.append(np.round(blended).astype(np.uint8))
        frames.append(stages[i + 1])
    return frames
