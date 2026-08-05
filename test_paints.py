"""Checks for the paint reference table and the nearest-tube match.

Run with:  python test_paints.py

The spec for this step called for three throwaway assertions, on the grounds
that there was no test infrastructure. There is now, so they live here beside
test_posterize.py and test_palette.py instead of being deleted.
"""

import numpy as np

from imaging import extract_palette, srgb_to_lab
from paints import PAINT_REFERENCE, nearest_paint


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. Identity: every tube must return itself at distance 0")

# This is the check that matters. It catches a broken conversion, a transposed
# axis, or a wrong distance function in one line, because there is exactly one
# right answer and it is knowable without judgment.
for paint in PAINT_REFERENCE:
    name, delta = nearest_paint(paint["lab"])
    assert name == paint["name"], f"{paint['name']} matched {name} instead of itself"
    assert delta < 1e-4, f"{paint['name']} matched itself at distance {delta}"
print(f"PASS: all {len(PAINT_REFERENCE)} tubes return themselves at distance 0")


# --------------------------------------------------------------------------
rule("2. The reference table is on the true CIE Lab scale")

lab = np.array([p["lab"] for p in PAINT_REFERENCE], dtype=np.float32)
print(f"  L ranges {lab[:, 0].min():6.2f} to {lab[:, 0].max():6.2f}")
print(f"  a ranges {lab[:, 1].min():6.2f} to {lab[:, 1].max():6.2f}")
print(f"  b ranges {lab[:, 2].min():6.2f} to {lab[:, 2].max():6.2f}")

# If these were the uint8 OpenCV scale, L would run toward 255 and a/b would
# be offset by +128 and never negative. Both being false is the check.
assert 0.0 <= lab[:, 0].min() and lab[:, 0].max() <= 100.0, "L is not on the 0-100 scale"
assert lab[:, 1].min() < 0 and lab[:, 2].min() < 0, "a/b never go negative, so this is the offset uint8 scale"
assert lab[:, 0].max() > 90, "no near-white tube, so the table is probably wrong"
print("PASS: L within 0-100 and a/b signed, so this matches imaging.srgb_to_lab")


# --------------------------------------------------------------------------
rule("3. Colors whose answer is knowable without judgment")

KNOWN = [
    ("pure white", (255, 255, 255), "Titanium White"),
    ("pure black", (0, 0, 0), "Ivory Black"),
    ("grass green", (74, 130, 50), "Chromium Oxide Green"),
    ("clear sky blue", (90, 140, 210), "Cerulean Blue"),
]
for label, rgb, expected in KNOWN:
    lab_value = srgb_to_lab(np.array([[rgb]], dtype=np.uint8)).reshape(3)
    name, delta = nearest_paint(lab_value)
    print(f"  {label:15} rgb{str(rgb):16} -> {name:22} dE {delta:5.1f}")
    assert name == expected, f"{label} matched {name}, expected {expected}"
print("PASS: a green photo returns a green tube, not titanium white for everything")


# --------------------------------------------------------------------------
rule("4. No neutral color gets a chromatic answer")

# The failure this catches: at near-zero chroma the nearest masstone is decided
# almost entirely by lightness, and whatever hue rides along is noise. Before
# Graphite Grey was added, a mid grey returned Chromium Oxide Green.
NEUTRAL_TUBES = {
    "Titanium White", "Graphite Grey", "Ivory Black", "Raw Umber", "Burnt Umber",
}
for level in range(10, 100, 10):
    grey = np.full((1, 1, 3), level * 255 // 100, dtype=np.uint8)
    lab_value = srgb_to_lab(grey).reshape(3)
    name, delta = nearest_paint(lab_value)
    chroma = float(np.hypot(lab_value[1], lab_value[2]))
    print(f"  grey L~{lab_value[0]:5.1f} chroma {chroma:4.1f} -> {name:22} dE {delta:5.1f}")
    assert name in NEUTRAL_TUBES, (
        f"a neutral at L {lab_value[0]:.1f} matched {name}, a chromatic tube. "
        "The reference palette is missing a neutral in that lightness range."
    )
print("PASS: neutrals match neutral tubes across the whole lightness range")


# --------------------------------------------------------------------------
rule("5. Coverage sums to 100 and matching is deterministic")

rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:300, 0:400]
photo = np.stack(
    [xx / 400 * 200 + 40, yy / 300 * 160 + 50, (xx + yy) / 700 * 180 + 30], axis=-1
)
photo[80:200, 100:260] *= 0.35
photo = np.clip(photo + rng.normal(0, 12, photo.shape), 0, 255).astype(np.uint8)

first = [
    (sw["hex"], sw["share"], *nearest_paint(sw["lab"]))
    for sw in extract_palette(photo)
]
total = sum(row[1] for row in first)
for hex_code, share, name, delta in first:
    print(f"  {hex_code}  {share*100:5.1f}%  ->  {name:22} dE {delta:5.1f}")
print(f"  coverage sums to {total*100:.2f}%")
assert abs(total - 1.0) < 1e-9, "coverage does not sum to 100"

for _ in range(5):
    again = [
        (sw["hex"], sw["share"], *nearest_paint(sw["lab"]))
        for sw in extract_palette(photo)
    ]
    assert again == first, "same photo gave different tube names"
print("PASS: sums to 100, and six runs give identical names and distances")


# --------------------------------------------------------------------------
rule("6. Distance range, which is why there is no per-item threshold")

distances = [row[3] for row in first]
print(f"  dE across this photo: {min(distances):.1f} to {max(distances):.1f}")
print("  The reference values are masstones and photo centroids are mostly")
print("  tints, so distances are large by construction. A per-item flag at")
print("  dE 12 would fire on nearly every swatch and therefore say nothing.")
print("  The panel is framed once, as closest tube, and shows the number.")

print()
print(f"ALL CHECKS PASSED  ({len(PAINT_REFERENCE)} tubes)")
