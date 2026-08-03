"""Rule 8 for the palette step.

Run with:  python test_palette.py

Self-contained: builds its own images, so it runs on a fresh clone with no
photo on disk. Asserts, so a wrong answer fails loudly.
"""

import numpy as np

from imaging import PALETTE_SIZE, downsample, extract_palette, lab_to_srgb, srgb_to_lab


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. Lab conversion, checked against colors with published answers")

# Standard sRGB (D65) to CIE Lab values. If the conversion path were the uint8
# one, white would come back near L=255 with a and b offset by +128 instead.
KNOWN = [
    ("white", (255, 255, 255), (100.00, 0.00, 0.00)),
    ("black", (0, 0, 0), (0.00, 0.00, 0.00)),
    ("red", (255, 0, 0), (53.24, 80.09, 67.20)),
    ("green", (0, 255, 0), (87.73, -86.18, 83.18)),
    ("blue", (0, 0, 255), (32.30, 79.19, -107.86)),
    ("mid gray", (128, 128, 128), (53.59, 0.00, 0.00)),
]

TOL = 1.5
for name, rgb, expected in KNOWN:
    got = srgb_to_lab(np.array([[rgb]], dtype=np.uint8)).reshape(3)
    delta = max(abs(g - e) for g, e in zip(got, expected))
    print(
        f"  {name:9} rgb{rgb!s:16} -> Lab "
        f"({got[0]:7.2f}, {got[1]:7.2f}, {got[2]:8.2f})   "
        f"expected ({expected[0]:6.2f}, {expected[1]:6.2f}, {expected[2]:7.2f})   "
        f"max diff {delta:.2f}"
    )
    assert delta < TOL, f"{name}: off by {delta:.2f}, wrong Lab scale or wrong dtype path"

print(f"PASS: all within {TOL} of published CIE Lab, so this is the float32 path")
print("      (the uint8 path would put white near L=255 and offset a/b by +128)")


# --------------------------------------------------------------------------
rule("2. Round trip sRGB -> Lab -> sRGB")

ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
color_ramp = np.stack([ramp, ramp[::-1], ramp.T], axis=-1)
back = lab_to_srgb(srgb_to_lab(color_ramp))
worst = int(np.abs(back.astype(int) - color_ramp.astype(int)).max())
print(f"  worst channel error over 256 colors: {worst}")
assert worst <= 2, f"round trip lost {worst} levels, expected 2 or fewer"
print("PASS: round trip is tight, but it is still lossy, which is why paint")
print("      matching uses the stored Lab and never re-derives it from RGB")


# --------------------------------------------------------------------------
rule("3. Reproduce small: 6 flat stripes of known width")

# Two stripes are deliberately the same width, to exercise the tie-break in
# the population sort.
STRIPES = [
    ((200, 30, 30), 30),
    ((30, 60, 200), 25),
    ((40, 150, 60), 20),
    ((230, 210, 60), 20),
    ((40, 40, 45), 15),
    ((240, 240, 235), 10),
]
assert sum(w for _, w in STRIPES) == 120

flat = np.zeros((120, 120, 3), dtype=np.uint8)
x = 0
for color, width in STRIPES:
    flat[:, x : x + width] = color
    x += width

palette = extract_palette(flat, k=6)

# Match each recovered swatch to its nearest planted color rather than by
# position. Two stripes share a width, so their order relative to each other
# is not something the population sort can decide, and asserting positionally
# would be asserting something the code never promised.
unclaimed = list(STRIPES)
print("  recovered            share    nearest planted color   share")
for sw in palette:
    best = min(
        unclaimed, key=lambda s: max(abs(a - b) for a, b in zip(s[0], sw["rgb"]))
    )
    err = max(abs(a - b) for a, b in zip(best[0], sw["rgb"]))
    expected_share = best[1] / 120.0
    print(
        f"  rgb{sw['rgb']!s:16} {sw['share']*100:5.2f}%   "
        f"rgb{best[0]!s:16} {expected_share*100:5.2f}%   "
        f"{sw['hex']}   err {err}"
    )
    assert err <= 2, f"recovered {sw['rgb']}, nothing planted is within 2 of it"
    assert abs(sw["share"] - expected_share) < 1e-9, "share does not match stripe width"
    unclaimed.remove(best)

assert not unclaimed, f"never recovered {unclaimed}"
print("PASS: every planted color recovered exactly once, with its exact share")

# The tie: stable sorting does not promise which of two equal-population
# clusters comes first. It promises the answer is the same every time, which
# is the property that actually matters.
tied = [sw["hex"] for sw in palette if abs(sw["share"] - 20 / 120.0) < 1e-9]
assert len(tied) == 2, "expected two clusters tied at 16.67%"
for _ in range(5):
    again = [sw["hex"] for sw in extract_palette(flat, k=6) if abs(sw["share"] - 20 / 120.0) < 1e-9]
    assert again == tied, f"tied pair reordered: {tied} then {again}"
print(f"PASS: the two clusters tied at 16.67% ({tied[0]}, {tied[1]}) resolve in the")
print("      same order on every run. Stable sort buys reproducibility, not a")
print("      guarantee about which one wins.")


# --------------------------------------------------------------------------
rule("4. Shares are a valid distribution, sorted descending")

shares = [sw["share"] for sw in palette]
print(f"  shares: {[round(s, 4) for s in shares]}")
print(f"  sum:    {sum(shares)}")
assert abs(sum(shares) - 1.0) < 1e-9, "shares must sum to 1"
assert shares == sorted(shares, reverse=True), "not sorted by population descending"
assert len(palette) == 6
print("PASS: sums to 1, ordered most-used first")


# --------------------------------------------------------------------------
rule("5. Determinism: same image, ten runs, identical swatches")

# A photo-like image: smooth gradient plus structure plus noise, so k-means
# has a genuinely ambiguous job rather than six obvious answers.
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:300, 0:400]
photo = np.stack(
    [
        (xx / 400 * 200 + 40),
        (yy / 300 * 160 + 50),
        ((xx + yy) / 700 * 180 + 30),
    ],
    axis=-1,
)
photo[80:200, 100:260] *= 0.35
photo = np.clip(photo + rng.normal(0, 12, photo.shape), 0, 255).astype(np.uint8)

first = extract_palette(photo)
identical = True
for i in range(9):
    again = extract_palette(photo)
    same = [s["rgb"] for s in again] == [s["rgb"] for s in first] and all(
        abs(a["share"] - b["share"]) < 1e-12 for a, b in zip(again, first)
    )
    identical &= same
    if not same:
        print(f"  run {i + 1} DIFFERS")

print(f"  palette: {[s['hex'] for s in first]}")
print(f"  shares:  {[round(s['share'] * 100, 1) for s in first]}")
print(f"  10 runs identical: {identical}")
assert identical, "same image gave different palettes, the seed is not holding"
print("PASS: reproducible. This is the check that never fires by accident,")
print("      because nobody thinks to upload the same photo twice.")


# --------------------------------------------------------------------------
rule("6. The seed is load-bearing (measured on a genuinely ambiguous image)")

# The image above has six obvious clusters, so every seed finds the same
# answer and it proves almost nothing. The dangerous case is data with no
# natural cluster structure, where initialization decides the outcome. A
# grayscale ramp is the extreme version: every pixel lies on the L axis with
# a = b = 0, so there is no correct way to cut it into six groups.
#
# This is not an academic case. A mostly desaturated photo (overcast light,
# dark clothing, grey rock) is close to the same near-1D distribution, which
# is why a real test photo drifted 41 Lab units between unseeded runs.
gray_ramp = np.repeat(
    (np.mgrid[0:300, 0:400][1] / 400 * 255).astype(np.uint8)[..., None], 3, axis=-1
)

anchor = extract_palette(gray_ramp, seed=0)
drifts = []
for seed in range(1, 6):
    p = extract_palette(gray_ramp, seed=seed)
    drifts.append(
        round(
            max(
                max(abs(a - b) for a, b in zip(x["lab"], y["lab"]))
                for x, y in zip(p, anchor)
            ),
            2,
        )
    )
print(f"  max Lab drift as the seed changes: {drifts}")
print("  A drift above ~5 Lab units is a visibly different color.")
assert max(drifts) > 10, (
    "changing the seed barely moved the centroids, so this image is too easy "
    "to demonstrate anything and the check is not testing what it claims"
)

# And the same seed still pins it down on exactly this hard image.
assert [s["lab"] for s in extract_palette(gray_ramp, seed=0)] == [
    s["lab"] for s in anchor
], "seed did not reproduce on the hard case"
print("PASS: seeds genuinely change the answer here, and seed=0 still reproduces.")
print("      Both halves matter. Without the first, the determinism check is")
print("      passing on an image where nothing could have gone wrong anyway.")


# --------------------------------------------------------------------------
rule("7. Downsampling caps the long edge and never upscales")

for shape, expected_long in [((3000, 4000), 200), ((150, 90), 150), ((200, 200), 200)]:
    src = np.zeros(shape + (3,), dtype=np.uint8)
    out = downsample(src, 200)
    print(f"  {shape} -> {out.shape[:2]}")
    assert max(out.shape[:2]) == expected_long, f"{shape} became {out.shape[:2]}"
print("PASS: large images shrink to 200, small images are left alone")

print()
print(f"ALL CHECKS PASSED  (k={PALETTE_SIZE})")
