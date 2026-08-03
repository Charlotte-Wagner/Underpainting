"""Image math for Underpainting.

Kept out of app.py on purpose: app.py runs Streamlit calls at import time, so
importing it from a plain terminal script is awkward. Everything here is pure
numpy in and numpy out, which means it can be run on a 4x4 array by hand.
"""

import numpy as np


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
