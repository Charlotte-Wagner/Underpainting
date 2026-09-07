"""Rule 8 for S9's new imaging.py functions: value_range, dominant_temperature,
load_rgb.

Run with:  python test_stats.py

Self-contained: builds its own arrays and images, so it runs on a fresh clone
with no photo on disk. Asserts, so a wrong answer fails loudly.
"""

from io import BytesIO

import numpy as np
from PIL import Image

from imaging import dominant_temperature, extract_palette, load_rgb, value_range


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. value_range on hand-built arrays")

CASES = [
    ("full range", np.array([[0, 128], [64, 255]], dtype=np.uint8), (0, 255)),
    ("compressed", np.array([[40, 90], [150, 210]], dtype=np.uint8), (40, 210)),
    ("flat", np.full((3, 3), 77, dtype=np.uint8), (77, 77)),
]

for name, arr, expected in CASES:
    got = value_range(arr)
    print(f"  {name:11} {arr.tolist()} -> {got}")
    assert got == expected, f"{name}: expected {expected}, got {got}"
    assert isinstance(got[0], int) and isinstance(got[1], int), "must be plain Python ints"
print("PASS: min/max returned exactly, not assumed to be 0 and 255")


# --------------------------------------------------------------------------
rule("2. dominant_temperature on a hand-built palette (weighted, not averaged)")

# Two swatches: a large cool sky (b=-20) and a small warm accent (b=+10).
# Plain mean would be -5. Share-weighted should land near the sky's value.
palette = [
    {"lab": (50.0, 0.0, -20.0), "share": 0.7},
    {"lab": (60.0, 0.0, 10.0), "share": 0.3},
]
expected = 0.7 * -20.0 + 0.3 * 10.0  # -11.0
got = dominant_temperature(palette)
print(f"  70% b=-20, 30% b=+10 -> {got} (expected {expected})")
assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"
plain_mean = sum(s["lab"][2] for s in palette) / len(palette)
print(f"  plain mean would have been {plain_mean}, weighting moved it toward the majority color")
assert abs(got - plain_mean) > 1.0, "weighting should differ from a plain mean here"
print("PASS: weighted by share, matches hand calculation")

rule("3. dominant_temperature sign on synthetic warm vs. cool images")

# Reuse extract_palette itself so this also exercises the real Lab pipeline,
# not just the weighting arithmetic in isolation.
warm_image = np.full((60, 60, 3), (230, 160, 40), dtype=np.uint8)  # orange
cool_image = np.full((60, 60, 3), (40, 90, 200), dtype=np.uint8)  # blue

warm_temp = dominant_temperature(extract_palette(warm_image, k=2))
cool_temp = dominant_temperature(extract_palette(cool_image, k=2))
print(f"  orange swatch -> {warm_temp:.2f}")
print(f"  blue swatch   -> {cool_temp:.2f}")
assert warm_temp > 0, "a flat orange image should read warm (positive b)"
assert cool_temp < 0, "a flat blue image should read cool (negative b)"
print("PASS: sign matches the axis the docstring claims (positive warm, negative cool)")


# --------------------------------------------------------------------------
rule("4. load_rgb: EXIF rotation is actually applied")

# Same synthetic-sideways-photo check as S3, run through the new shared
# function instead of app.py's inline steps.
#
# 80x40 rather than the 40x20 this used through S19, because load_rgb now refuses
# anything under MIN_SOURCE_DIMENSION on its longest side. Verified by mutation
# that the size is incidental: with exif_transpose removed, the shape assertion
# below fails at 40x20 and at 80x40 alike.
base = Image.new("RGB", (80, 40), (10, 10, 10))
base.putpixel((0, 0), (250, 0, 0))  # distinct marker in the top-left corner
exif = base.getexif()
exif[0x0112] = 6  # orientation tag: rotate 90 CW to display correctly
buf = BytesIO()
base.save(buf, format="JPEG", exif=exif)

rotated = load_rgb(buf.getvalue(), max_dimension=1200)
plain = np.array(base)  # no exif applied, for comparison
print(f"  no-rotation shape: {plain.shape}   load_rgb shape: {rotated.shape}")
assert rotated.shape[:2] != plain.shape[:2], "orientation tag 6 should swap width/height"


def corner_redness(image, corner, size=3):
    """How far the red channel leads the other two, averaged over a corner patch.

    A patch rather than the single marker pixel, and a comparison rather than an
    equality test, because the fixture is a JPEG. Lossy compression smears one
    bright pixel on a flat ground across its whole 8x8 block: the marker is
    written as (250, 0, 0) and comes back around (31, 11, 17). Any assertion
    naming the exact value is therefore true no matter what rotation happened,
    which is precisely how the check this replaced managed to never fail.
    """
    rows = slice(0, size) if corner[0] == "top" else slice(-size, None)
    cols = slice(0, size) if corner[1] == "left" else slice(-size, None)
    patch = image[rows, cols].reshape(-1, 3).astype(float)
    return float(np.mean(patch[:, 0] - (patch[:, 1] + patch[:, 2]) / 2.0))


# Orientation 6 rotates 90 degrees clockwise, which sends the pixel at (0, 0) to
# the top-right. Checking where the marker landed, rather than only that the axes
# swapped, is what makes this catch a rotation applied in the wrong direction:
# ROTATE_90 the other way also swaps the axes and passes the assertion above, but
# puts the marker bottom-left.
#
# The suite was not blind to that case before, but only by accident: check 5's
# fixture is non-square, so a wrong-direction rotation tripped its resize assertion
# instead. A rotation bug reported as a resize failure, two sections below the check
# named for rotation, is a worse way to find out.
CORNERS = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
measured = {c: corner_redness(rotated, c) for c in CORNERS}
for (vertical, horizontal), value in measured.items():
    print(f"  {vertical + '-' + horizontal:<13} redness {value:+6.1f}")

marker = max(measured, key=measured.get)
runner_up = sorted(measured.values())[-2]
assert marker == ("top", "right"), (
    f"orientation 6 should put the marker top-right, found it {marker[0]}-{marker[1]}"
)
assert measured[marker] - runner_up > 5.0, (
    "the marker corner should stand clearly above the rest, "
    f"got {measured[marker]:.1f} against {runner_up:.1f}"
)
print("PASS: load_rgb applies EXIF rotation, and applies it in the right direction")

rule("5. load_rgb: resizes down, never up")

big = Image.new("RGB", (3000, 1500), (5, 5, 5))
buf = BytesIO()
big.save(buf, format="JPEG")
out = load_rgb(buf.getvalue(), max_dimension=1200)
print(f"  3000x1500 -> {out.shape[:2]}")
assert max(out.shape[:2]) == 1200, f"expected long edge 1200, got {max(out.shape[:2])}"

small = Image.new("RGB", (80, 40), (5, 5, 5))
buf = BytesIO()
small.save(buf, format="JPEG")
out = load_rgb(buf.getvalue(), max_dimension=1200)
print(f"  80x40 -> {out.shape[:2]}")
assert out.shape[:2] == (40, 80), "small image should be left alone, not upscaled"
print("PASS: caps large photos, leaves small ones untouched")

print()
print("ALL CHECKS PASSED")
