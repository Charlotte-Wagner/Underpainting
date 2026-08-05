"""Rule 8 for the GIF-encoding step.

Run with:  python test_gif.py

The trap this guards against (see debug-log.md, diamond-4): Pillow quantizes
each GIF frame separately by default, so consecutive frames get slightly
different palettes and the animation flickers on loop. That is invisible in
a screenshot, so this file inspects the encoded bytes directly rather than
looking at a rendered frame. Asserts, so a wrong answer fails loudly.
"""

from io import BytesIO

import numpy as np
from PIL import Image

from gif import encode_gif, frame_durations


def rule(title):
    print()
    print(title)
    print("-" * len(title))


def local_color_table_flags(data):
    """Whether the GIF file's global color table exists, and which frames
    also carry their own local one. A local table on any frame after the
    first is the literal mechanism of the flicker bug: that frame is not
    using the shared table, so its colors were quantized independently.
    """
    assert data[:6] in (b"GIF87a", b"GIF89a"), "not a GIF"
    packed = data[10]
    gct_flag = bool(packed & 0x80)
    gct_size = 3 * (2 ** ((packed & 0x07) + 1)) if gct_flag else 0
    pos = 13 + gct_size
    flags = []
    while pos < len(data):
        block_type = data[pos]
        if block_type == 0x3B:  # trailer
            break
        elif block_type == 0x21:  # extension block
            pos += 2
            while True:
                size = data[pos]
                pos += 1
                if size == 0:
                    break
                pos += size
        elif block_type == 0x2C:  # image descriptor
            img_packed = data[pos + 9]
            lct_flag = bool(img_packed & 0x80)
            lct_size = 3 * (2 ** ((img_packed & 0x07) + 1)) if lct_flag else 0
            flags.append(lct_flag)
            pos += 10 + lct_size + 1  # descriptor + local table + LZW min code size
            while True:
                size = data[pos]
                pos += 1
                if size == 0:
                    break
                pos += size
        else:
            raise ValueError(f"unrecognized GIF block {block_type:#x} at byte {pos}")
    return gct_flag, flags


# --------------------------------------------------------------------------
rule("1. frame_durations matches cross_fade_frames's own frame order")

# imaging.cross_fade_frames emits: stage, fade x n, stage, fade x n, ..., stage.
# This has to produce the same shape or the durations line up with the wrong
# frames -- a stage frame reading as a 45ms flash instead of a 650ms pause.
for stage_count, fade_frames in [(2, 3), (4, 6), (4, 0)]:
    durations = frame_durations(stage_count, fade_frames)
    expected_len = stage_count + (stage_count - 1) * fade_frames
    print(f"  stages={stage_count} fade_frames={fade_frames}: {durations}")
    assert len(durations) == expected_len
    stage_positions = [0] + [
        1 + i * (fade_frames + 1) + fade_frames for i in range(stage_count - 1)
    ]
    for i, d in enumerate(durations):
        if i in stage_positions:
            assert d == 650, f"position {i} should be a stage hold, got {d}ms"
        else:
            assert d == 45, f"position {i} should be a fade frame, got {d}ms"
print("PASS: stage positions get the long hold, fade positions get the short one")


# --------------------------------------------------------------------------
rule("2. encode_gif rejects mismatched frames/durations lengths")

frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
try:
    encode_gif(frames, [100, 100])
except ValueError as e:
    print(f"PASS: ValueError({e})")
else:
    raise AssertionError("mismatched lengths should have raised")


# --------------------------------------------------------------------------
rule("3. encode_gif produces one shared color table, no per-frame tables")

red = np.full((6, 6, 3), (220, 20, 20), dtype=np.uint8)
mid1 = np.full((6, 6, 3), (165, 20, 87), dtype=np.uint8)
mid2 = np.full((6, 6, 3), (110, 20, 154), dtype=np.uint8)
mid3 = np.full((6, 6, 3), (55, 20, 220), dtype=np.uint8)
blue = np.full((6, 6, 3), (0, 20, 255), dtype=np.uint8)
frames = [red, mid1, mid2, mid3, blue]
durations = frame_durations(stage_count=2, fade_frames=3)

data = encode_gif(frames, durations)
gct_flag, lct_flags = local_color_table_flags(data)
print(f"  global color table present: {gct_flag}")
print(f"  per-frame local color table flags: {lct_flags}")
assert gct_flag, "no global color table at all"
assert not any(lct_flags), (
    f"frame(s) {[i for i, f in enumerate(lct_flags) if f]} carry their own local "
    "color table -- those frames were quantized independently and will flicker"
)
print(f"  file size: {len(data)} bytes for {len(frames)} frames")
print("PASS: every frame after the first relies on the one shared global table")


# --------------------------------------------------------------------------
rule("4. Negative control: the naive per-frame quantize actually does flicker")

# This is the check that proves check 3 could have failed. Converting each
# frame to \"P\" mode independently -- no shared reference palette, which is
# what Image.convert(\"P\") and a bare Image.quantize() with no palette=
# argument both do -- is the exact naive approach the build plan warns
# about. If this negative control did NOT show local tables, check 3 would
# be proving nothing.
naive_frames = [Image.fromarray(f).convert("P") for f in frames]
buffer = BytesIO()
naive_frames[0].save(
    buffer, format="GIF", save_all=True, append_images=naive_frames[1:], duration=durations, loop=0
)
naive_gct, naive_lct = local_color_table_flags(buffer.getvalue())
print(f"  naive per-frame quantize -- local color table flags: {naive_lct}")
assert any(naive_lct), (
    "expected the naive per-frame quantize to write local color tables; if it "
    "didn't, check 3 above is not distinguishing a fix from a bug"
)
print("PASS: the naive path really does flicker, so the shared-palette fix in")
print("      check 3 is fixing a real thing, not passing on a test that could")
print("      never fail")

print()
print("ALL CHECKS PASSED")
