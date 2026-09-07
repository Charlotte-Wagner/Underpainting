"""Rule 8: reproduce small.

Run with:  python test_load_guards.py

Checks the two size guards in load_rgb, and checks that they run against the
header rather than against a decoded image, which is the only thing that makes
the larger of them worth having. Asserts, so a guard that quietly stops
guarding fails loudly instead of printing something plausible.

Every image here is built in memory from a tiny array. Nothing is read from
disk and nothing needs the API key.
"""

import io
import resource
import warnings

import numpy as np
from PIL import Image

from imaging import (
    MAX_SOURCE_PIXELS,
    MIN_SOURCE_DIMENSION,
    UnusableImageSize,
    load_rgb,
)

MAX_DIMENSION = 1200

# Pillow warns about the deliberately oversized image below. The warning is the
# thing being tested around, not a surprise, so it does not need printing.
warnings.simplefilter("ignore")


def rule(title):
    print()
    print(title)
    print("-" * len(title))


def png_bytes(width, height, channels=3):
    """A PNG of the given size, compressed hard so a huge one stays a small file.

    Solid black on purpose: it is the cheapest thing to build at 13300x13300 and
    the guards being tested never look at pixel values, only at dimensions.
    """
    shape = (height, width) if channels == 1 else (height, width, channels)
    buffer = io.BytesIO()
    Image.fromarray(np.zeros(shape, np.uint8)).save(
        buffer, format="PNG", compress_level=9
    )
    return buffer.getvalue()


def peak_mib():
    """Peak RSS of this process in MiB. macOS reports ru_maxrss in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


rule("1. The constants being enforced")

print(f"MAX_SOURCE_PIXELS   = {MAX_SOURCE_PIXELS:,} ({MAX_SOURCE_PIXELS / 1e6:.0f} MP)")
print(f"MIN_SOURCE_DIMENSION = {MIN_SOURCE_DIMENSION} px on the longest side")
assert MAX_SOURCE_PIXELS > 40_000_000, "a 48MP phone photo has to still be accepted"
assert MIN_SOURCE_DIMENSION >= 6, "below 6 pixels extract_palette cannot find 6 clusters"
print("PASS: both constants are in the range the app is designed around")


rule("2. Ordinary photos are accepted, and come back as (H, W, 3) uint8")

for width, height in [(64, 64), (1200, 800), (800, 1200), (4000, 3000)]:
    rgb = load_rgb(png_bytes(width, height), MAX_DIMENSION)
    longest = max(rgb.shape[:2])
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected 3 channels, got {rgb.shape}"
    assert rgb.dtype == np.uint8, f"expected uint8, got {rgb.dtype}"
    assert longest <= MAX_DIMENSION, f"{longest} exceeds the {MAX_DIMENSION} cap"
    print(f"PASS: {width}x{height} -> {rgb.shape}")


rule("3. The floor, which is where the pipeline used to raise an unhandled error")

# extract_palette needs PALETTE_SIZE samples and raises below that. Before this
# guard existed the ValueError escaped prepare_photo's UnreadableImage handler,
# so a 2x2 PNG put a Python traceback on a public page.
for width, height in [(1, 1), (2, 2), (16, 16), (MIN_SOURCE_DIMENSION - 1,) * 2]:
    try:
        load_rgb(png_bytes(width, height), MAX_DIMENSION)
    except UnusableImageSize as e:
        assert str(width) in str(e), "the message has to name the size it refused"
        print(f"PASS: {width}x{height} refused")
    else:
        raise AssertionError(f"{width}x{height} should have been refused")

# One pixel over the line has to be accepted, or the boundary is off by one.
edge = load_rgb(png_bytes(MIN_SOURCE_DIMENSION, MIN_SOURCE_DIMENSION), MAX_DIMENSION)
print(f"PASS: {MIN_SOURCE_DIMENSION}x{MIN_SOURCE_DIMENSION} accepted -> {edge.shape}")


rule("4. The ceiling, checked at both sides of the boundary")

over = int(MAX_SOURCE_PIXELS**0.5) + 200
try:
    load_rgb(png_bytes(over, over), MAX_DIMENSION)
except UnusableImageSize as e:
    print(f"PASS: {over}x{over} ({over * over / 1e6:.0f} MP) refused")
    print(f"      {e}")
else:
    raise AssertionError(f"{over}x{over} should have been refused")

under = int(MAX_SOURCE_PIXELS**0.5) - 200
accepted = load_rgb(png_bytes(under, under), MAX_DIMENSION)
print(f"PASS: {under}x{under} ({under * under / 1e6:.0f} MP) accepted -> {accepted.shape}")


rule("5. The ceiling is enforced before the decode, not after it")

# This is the whole reason the ceiling exists. Pillow's own decompression-bomb
# check only raises above twice its MAX_IMAGE_PIXELS, so a 13300x13300 file
# sits just under it, decodes, and took peak RSS to 932MiB when measured. A
# guard that ran after the decode would report the same error and prevent
# nothing. Measured here rather than asserted on the exception alone.
bomb = png_bytes(13300, 13300, channels=1)
print(f"a 13300x13300 PNG is {len(bomb) / 1048576:.2f} MiB on disk (177 MP)")
print(f"Pillow's own limit is {Image.MAX_IMAGE_PIXELS:,}, and it only raises above "
      f"{2 * Image.MAX_IMAGE_PIXELS:,}")

before = peak_mib()
try:
    load_rgb(bomb, MAX_DIMENSION)
except UnusableImageSize:
    pass
else:
    raise AssertionError("the 177 MP image should have been refused")
grew = peak_mib() - before

# 177 MP decodes to roughly 620MiB. Refusing it should cost nothing measurable;
# 100MiB is a loose bound that still fails if the decode ever happens again.
print(f"PASS: refused, and peak RSS grew {grew:.0f} MiB while doing it")
assert grew < 100, f"peak RSS grew {grew:.0f} MiB, so it decoded before refusing"


rule("6. A file that is not an image is still someone else's error")

# The size guards must not swallow the decode failure. Bytes that are not an
# image have to keep raising something app.py turns into "try a JPEG, PNG, or
# HEIC photo", not the size message, which would be false and unactionable.
for name, data in [("empty file", b""), ("text file", b"not an image\n" * 40)]:
    try:
        load_rgb(data, MAX_DIMENSION)
    except UnusableImageSize:
        raise AssertionError(f"{name} was reported as a size problem, which it is not")
    except Exception as e:
        print(f"PASS: {name} raised {type(e).__name__}, which prepare_photo wraps")
    else:
        raise AssertionError(f"{name} should not have decoded")


print()
print("ALL CHECKS PASSED")
