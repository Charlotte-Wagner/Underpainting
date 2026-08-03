"""Rule 8: reproduce small.

Run with:  python test_posterize.py

Checks posterize on a 4x4 array you can verify by eye, then on a full 0-255
ramp where every band is guaranteed to appear. Asserts, so a wrong answer
fails loudly instead of printing something plausible.
"""

import numpy as np
from PIL import Image, ImageOps

from imaging import posterize, posterize_output_values

# 16 numbers chosen by hand. Rows 1-2 spread across the range; row 3 straddles
# both n=3 decision boundaries (63.75 and 191.25) on purpose; row 4 is an even
# ramp.
SMALL = np.array(
    [
        [0, 60, 70, 127],
        [128, 190, 192, 255],
        [63, 64, 191, 192],
        [32, 96, 160, 224],
    ],
    dtype=np.uint8,
)


def rule(title):
    print()
    print(title)
    print("-" * len(title))


rule("1. The actual output values")

for n in (3, 5):
    step = 255.0 / (n - 1)
    exact = np.arange(n) * step
    print(f"n={n}: step = 255/{n - 1} = {step}")
    print(f"      exact  {[round(v, 3) for v in exact.tolist()]}")
    print(f"      uint8  {posterize_output_values(n).tolist()}")

rule("2. Decision boundaries (round-to-nearest splits at the midpoints)")

for n in (3, 5):
    step = 255.0 / (n - 1)
    cuts = [(k + 0.5) * step for k in range(n - 1)]
    print(f"n={n}: a pixel moves up a band at {[round(c, 3) for c in cuts]}")

rule("3. No integer 0-255 lands exactly on a rounding tie")

for n in (3, 5):
    step = 255.0 / (n - 1)
    q = np.arange(256, dtype=np.float32) / step
    ties = np.arange(256)[np.abs(q - np.floor(q) - 0.5) < 1e-6]
    print(f"n={n}: tied inputs = {ties.tolist()}")
    assert ties.size == 0, "a tie means np.round's half-to-even rule decides the band"
print("PASS: every input rounds unambiguously, so no band boundary is arbitrary")

rule("4. The 4x4, n=3")

print("input:")
print(SMALL)
got3 = posterize(SMALL, 3)
print("output:")
print(got3)

expected3 = np.array(
    [
        [0, 0, 128, 128],
        [128, 128, 255, 255],
        [0, 128, 128, 255],
        [0, 128, 128, 255],
    ],
    dtype=np.uint8,
)
assert np.array_equal(got3, expected3), f"expected\n{expected3}\ngot\n{got3}"
print("PASS: matches the hand-computed answer")
print("      note row 3: 63 -> 0 but 64 -> 128, and 191 -> 128 but 192 -> 255")
print("      those are the 63.75 and 191.25 boundaries, visible in four numbers")

rule("5. The 4x4, n=5")

got5 = posterize(SMALL, 5)
print("output:")
print(got5)

expected5 = np.array(
    [
        [0, 64, 64, 128],
        [128, 191, 191, 255],
        [64, 64, 191, 191],
        [64, 128, 191, 255],
    ],
    dtype=np.uint8,
)
assert np.array_equal(got5, expected5), f"expected\n{expected5}\ngot\n{got5}"
print("PASS: matches the hand-computed answer")

rule("6. np.unique on a full 0-255 ramp: exactly n, first 0, last 255")

ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)

for n in (3, 5):
    values = np.unique(posterize(ramp, n))
    print(f"n={n}: np.unique -> {values.tolist()}   (count {values.size})")
    assert values.size == n, f"expected {n} distinct values, got {values.size}"
    assert values[0] == 0, f"darkest value is {values[0]}, not true black"
    assert values[-1] == 255, f"lightest value is {values[-1]}, not true white"
print("PASS: exactly n values, starting at 0, ending at 255")

rule("7. Every n from 2 to 32, same three properties")

for n in range(2, 33):
    values = np.unique(posterize(ramp, n))
    assert values.size == n, f"n={n} produced {values.size} values"
    assert values[0] == 0 and values[-1] == 255, f"n={n} endpoints wrong"
print("PASS: n=2..32 all give exactly n values anchored at 0 and 255")

rule("8. The rejected alternative: Pillow's ImageOps.posterize")

for bits in (1, 2, 3):
    pil_out = np.array(ImageOps.posterize(Image.fromarray(ramp), bits))
    values = np.unique(pil_out)
    print(f"bits={bits}: {values.size} values, lightest = {values[-1]}   {values.tolist()[:6]}")
print("Takes bits, not levels, so 3 and 5 are not askable for at all.")
print("And the lightest value is never 255. That is the bug this session is about.")

rule("9. n_levels < 2 is rejected rather than dividing by zero")

try:
    posterize(SMALL, 1)
except ValueError as e:
    print(f"PASS: ValueError({e})")
else:
    raise AssertionError("n_levels=1 should have raised")

print()
print("ALL CHECKS PASSED")
