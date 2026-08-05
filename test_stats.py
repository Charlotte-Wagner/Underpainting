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
base = Image.new("RGB", (40, 20), (10, 10, 10))
base.putpixel((0, 0), (250, 0, 0))  # distinct marker in the top-left corner
exif = base.getexif()
exif[0x0112] = 6  # orientation tag: rotate 90 CW to display correctly
buf = BytesIO()
base.save(buf, format="JPEG", exif=exif)

rotated = load_rgb(buf.getvalue(), max_dimension=1200)
plain = np.array(base)  # no exif applied, for comparison
print(f"  no-rotation shape: {plain.shape}   load_rgb shape: {rotated.shape}")
assert rotated.shape[:2] != plain.shape[:2], "orientation tag 6 should swap width/height"
assert tuple(rotated[0, 0]) != (250, 0, 0), "top-left marker pixel should have moved"
print("PASS: load_rgb applies EXIF rotation, same check S3 did by hand")

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
