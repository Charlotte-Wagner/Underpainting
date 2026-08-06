"""Rule 8 for the stage-generation step.

Run with:  python test_stages.py

Self-contained: builds its own images, so it runs on a fresh clone with no
photo on disk. Asserts, so a wrong answer fails loudly instead of printing
something plausible.
"""

import numpy as np

from imaging import (
    TONED_GROUND,
    anchor_masses,
    build_stages,
    cross_fade_frames,
    line_drawing,
    posterize_output_values,
    recolor_to_palette,
    srgb_to_lab,
    value_block_in,
)


def swatch(rgb_tuple):
    """A palette entry for an exact color, built the way extract_palette would."""
    lab = srgb_to_lab(np.array([[rgb_tuple]], dtype=np.uint8)).reshape(3)
    return {"lab": tuple(float(v) for v in lab), "rgb": rgb_tuple}


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

palette = [swatch(RED), swatch(BLUE)]

recolored = recolor_to_palette(board, palette)
distinct = set(map(tuple, recolored.reshape(-1, 3).tolist()))
print(f"  distinct colors in output: {distinct}")
assert distinct == {RED, BLUE}, f"a third color appeared: {distinct - {RED, BLUE}}"
assert np.array_equal(recolored, board), "recolor changed pixels that were already exact palette colors"
print("PASS: every pixel snapped to its exact planted color, columns stay sharp")


# --------------------------------------------------------------------------
rule("3. anchor_masses: the two end bands get paint, the middle band stays ground")

# Three vertical bars, one per value band, each already an exact palette
# color. Chosen so posterize's 3-level banding puts one bar in each band:
# grayscale lands near 20, 122, and 231, and the band boundaries sit at
# multiples of 127.5.
DARK = (20, 20, 30)
MID = (128, 120, 110)
LIGHT = (235, 230, 225)
bars = np.zeros((30, 30, 3), dtype=np.uint8)
bars[:, 0:10] = DARK
bars[:, 10:20] = MID
bars[:, 20:30] = LIGHT

anchored = anchor_masses(bars, [swatch(DARK), swatch(MID), swatch(LIGHT)])
columns = [tuple(anchored[15, x].tolist()) for x in (5, 15, 25)]
print(f"  darkest bar -> {columns[0]}, middle bar -> {columns[1]}, lightest bar -> {columns[2]}")

assert columns[0] == DARK, "the darkest band should be painted its own palette color"
assert columns[2] == LIGHT, "the lightest band should be painted its own palette color"
assert columns[1] == (TONED_GROUND,) * 3, (
    "the middle band should be left as bare ground -- if it is painted, stage 2 "
    "and stage 3 are the same picture and the midtone step shows nothing arriving"
)
print("PASS: both anchors placed in palette color, the midtone band untouched")


# --------------------------------------------------------------------------
rule("4. line_drawing: black on the toned ground, no frame border, same answer twice")

# A dark rectangle on a light field: one shape, one real boundary, and a
# long enough one to survive the fragment-length filter at this size.
field = np.full((200, 200, 3), 220, dtype=np.uint8)
field[40:160, 40:160] = 30
drawing = line_drawing(field, [swatch((30, 30, 30)), swatch((220, 220, 220))])

distinct_line_values = sorted(set(drawing.reshape(-1).tolist()))
margin = max(2, int(round(200 * 0.004)))
border = np.concatenate([
    drawing[:margin].reshape(-1), drawing[-margin:].reshape(-1),
    drawing[:, :margin].reshape(-1), drawing[:, -margin:].reshape(-1),
])
ink_fraction = float((drawing[:, :, 0] < TONED_GROUND).mean())

print(f"  distinct values: {distinct_line_values}")
print(f"  ink coverage: {ink_fraction * 100:.1f}%")
print(f"  frame margin ({margin}px) all ground: {bool((border == TONED_GROUND).all())}")

assert distinct_line_values == [0, TONED_GROUND], (
    "the drawing should be black lines on the toned ground and nothing else"
)
assert (border == TONED_GROUND).all(), (
    "a line was drawn in the frame margin -- the photo's own border is a boundary "
    "of every region touching it, so without the margin every photo gets a box "
    "drawn around it"
)
# Two-sided: an empty drawing and an all-black one both pass a "does it run"
# check, and both are useless.
assert 0.001 < ink_fraction < 0.5, f"ink coverage of {ink_fraction:.3f} is not a drawing"
assert np.array_equal(
    drawing, line_drawing(field, [swatch((30, 30, 30)), swatch((220, 220, 220))])
), "two runs on the same input gave different drawings"
print("PASS: black on the ground, the frame is not a line, and the result is deterministic")


# --------------------------------------------------------------------------
rule("5. build_stages: all four computed from the untouched source, not chained")

# Same checkerboard, so stage 3 (recolor) is checked against the same
# sharp-edge standard as check 2, but this time through build_stages, where
# a chaining bug (recoloring an earlier stage instead of the source) would
# actually show up.
stages = build_stages(board, palette)
assert len(stages) == 4
for s in stages:
    assert s.shape == board.shape

drawing_stage, anchor_stage, color_stage, final_stage = stages

print("  stage 4 (full detail) byte-identical to source:", np.array_equal(final_stage, board))
assert np.array_equal(final_stage, board), "stage 4 drifted from the actual upload"

color_distinct = set(map(tuple, color_stage.reshape(-1, 3).tolist()))
print(f"  stage 3 (midtones) distinct colors: {color_distinct}")
assert color_distinct == {RED, BLUE}, (
    "stage 3 introduced a color outside the planted pair -- it was likely "
    "computed from an earlier stage instead of the source"
)

# Stage 2 may only ever paint with the palette or leave canvas bare. Any
# other color means it stopped being the palette panel's own colors, which
# is the whole reason the color stages use the palette rather than a fresh
# quantization of their own.
anchor_distinct = set(map(tuple, anchor_stage.reshape(-1, 3).tolist()))
allowed = {RED, BLUE, (TONED_GROUND,) * 3}
print(f"  stage 2 (darks and lights) distinct colors: {anchor_distinct}")
assert anchor_distinct <= allowed, f"stage 2 painted with {anchor_distinct - allowed}"

drawing_values = sorted(set(drawing_stage.reshape(-1).tolist()))
print(f"  stage 1 (drawing) distinct values: {drawing_values}")
assert drawing_values == [0, TONED_GROUND], "stage 1 is not a drawing on the toned ground"

print("PASS: stage 3 is still sharp-edged and stage 4 is exactly the source,")
print("      which is what parallel computation looks like; a chained version")
print("      would have softened stage 3's edges or drifted stage 4 off the photo")


# --------------------------------------------------------------------------
rule("6. cross_fade_frames: length, exact endpoints, hand-computed midpoints")

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
