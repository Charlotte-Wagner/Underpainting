"""Rule 8 for the stage-generation step.

Run with:  python test_stages.py

Self-contained: builds its own images, so it runs on a fresh clone with no
photo on disk. Asserts, so a wrong answer fails loudly instead of printing
something plausible.
"""

import numpy as np

from imaging import (
    build_stages,
    cross_fade_frames,
    posterize,
    posterize_output_values,
    recolor_to_palette,
    soften,
    srgb_to_lab,
    value_block_in,
)


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. value_block_in is posterize on grayscale, broadcast to 3 channels")

ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
ramp_rgb = np.stack([ramp, ramp, ramp], axis=-1)  # already grayscale, so cvtColor is a no-op
out = value_block_in(ramp_rgb, n_levels=3)
expected_values = set(posterize_output_values(3).tolist())

print(f"  shape: {out.shape}")
print(f"  distinct values: {sorted(set(out.reshape(-1).tolist()))}")
assert out.shape == ramp_rgb.shape
assert set(out.reshape(-1).tolist()) == expected_values
assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])
print("PASS: same value set posterize(gray, 3) gives, replicated across R, G, B equally")


# --------------------------------------------------------------------------
rule("2. recolor_to_palette recovers exact planted colors, no third color at edges")

# A sharp checkerboard of two exact palette colors. If recolor ran on a
# blurred copy instead of the untouched source, edge pixels would land on a
# blend and round to something other than either planted color.
RED = (200, 30, 30)
BLUE = (30, 60, 200)
board = np.zeros((40, 40, 3), dtype=np.uint8)
board[:, ::2] = RED
board[:, 1::2] = BLUE

palette = [
    {"lab": tuple(float(v) for v in srgb_to_lab(np.array([[RED]], dtype=np.uint8)).reshape(3)), "rgb": RED},
    {"lab": tuple(float(v) for v in srgb_to_lab(np.array([[BLUE]], dtype=np.uint8)).reshape(3)), "rgb": BLUE},
]

recolored = recolor_to_palette(board, palette)
distinct = set(map(tuple, recolored.reshape(-1, 3).tolist()))
print(f"  distinct colors in output: {distinct}")
assert distinct == {RED, BLUE}, f"a third color appeared: {distinct - {RED, BLUE}}"
assert np.array_equal(recolored, board), "recolor changed pixels that were already exact palette colors"
print("PASS: every pixel snapped to its exact planted color, columns stay sharp")


# --------------------------------------------------------------------------
rule("3. soften: blur radius is a fraction of the long edge, not a fixed pixel count")

small = np.zeros((10, 10, 3), dtype=np.uint8)
small[5, 5] = 255
big = np.zeros((100, 100, 3), dtype=np.uint8)
big[50, 50] = 255
wide = np.zeros((20, 200, 3), dtype=np.uint8)  # non-square: long edge is width
wide[10, 100] = 255

spread_small = int(np.count_nonzero(soften(small)[:, :, 0]))
spread_big = int(np.count_nonzero(soften(big)[:, :, 0]))
spread_wide = int(np.count_nonzero(soften(wide)[:, :, 0]))

print(f"  10x10 (long edge 10):   {spread_small} pixels touched")
print(f"  100x100 (long edge 100): {spread_big} pixels touched")
print(f"  20x200 (long edge 200):  {spread_wide} pixels touched")

assert spread_big > spread_small, "a 10x larger image should blur over more pixels, not the same amount"
# The 20x200 image has a longer edge than the 100x100 one (200 vs 100), so
# its blur radius, and therefore its spread, should be larger too. This is
# the check that would fail if the radius were keyed off height, or off a
# fixed pixel count, instead of max(height, width).
assert spread_wide > spread_big, "long edge is 200 here, wider than the 100x100 case, but spread did not grow"
print("PASS: spread grows with the long edge, including the non-square case")


# --------------------------------------------------------------------------
rule("4. build_stages: all four computed from the untouched source, not chained")

# Same checkerboard, so stage 2 (recolor) is checked against the same
# sharp-edge standard as check 2, but this time through build_stages, where
# a chaining bug (recoloring the blurred stage instead of the source) would
# actually show up.
stages = build_stages(board, palette)
assert len(stages) == 4
for s in stages:
    assert s.shape == board.shape

value_stage, color_stage, soft_stage, final_stage = stages

print("  stage 4 (full detail) byte-identical to source:", np.array_equal(final_stage, board))
assert np.array_equal(final_stage, board), "stage 4 drifted from the actual upload"

color_distinct = set(map(tuple, color_stage.reshape(-1, 3).tolist()))
print(f"  stage 2 (color masses) distinct colors: {color_distinct}")
assert color_distinct == {RED, BLUE}, (
    "stage 2 introduced a color outside the planted pair -- it was likely "
    "computed from a blurred or posterized stage instead of the source"
)

value_values = set(value_stage.reshape(-1).tolist())
print(f"  stage 1 (values) distinct values: {sorted(value_values)}")
assert value_values.issubset(expected_values), (
    f"stage 1 produced {value_values - expected_values}, outside the n=3 "
    "posterize levels -- it did not come from posterizing the source's own grayscale"
)
print("PASS: stage 2 is still sharp-edged and stage 4 is exactly the source,")
print("      which is what parallel computation looks like; a chained version")
print("      would have blurred stage 2's edges or drifted stage 4 off the photo")


# --------------------------------------------------------------------------
rule("5. cross_fade_frames: length, exact endpoints, hand-computed midpoints")

a = np.zeros((2, 2, 3), dtype=np.uint8)
b = np.full((2, 2, 3), 90, dtype=np.uint8)
c = np.full((2, 2, 3), 210, dtype=np.uint8)

for fade_frames in (0, 1, 3, 6):
    frames = cross_fade_frames([a, b, c], fade_frames)
    expected_len = 3 + 2 * fade_frames
    print(f"  fade_frames={fade_frames}: {len(frames)} frames (expected {expected_len})")
    assert len(frames) == expected_len
    assert np.array_equal(frames[0], a)
    assert np.array_equal(frames[-1], c)

frames = cross_fade_frames([a, b], fade_frames=3)
# t = 1/4, 2/4, 3/4 of the 0 -> 90 step
expected_mid = [round(90 * t) for t in (0.25, 0.5, 0.75)]
got_mid = [int(f[0, 0, 0]) for f in frames[1:4]]
print(f"  fade_frames=3, 0 -> 90: intermediate values {got_mid} (expected {expected_mid})")
assert got_mid == expected_mid
print("PASS: frame count matches the formula, endpoints are exact, and the")
print("      interpolated values match hand computation, not just monotonicity")

print()
print("ALL CHECKS PASSED")
