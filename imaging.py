"""Image math for Underpainting.

Kept out of app.py on purpose: app.py runs Streamlit calls at import time, so
importing it from a plain terminal script is awkward. Everything here is pure
numpy in and numpy out, which means it can be run on a 4x4 array by hand.
"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps

# Clustering settings. Hardcoded per cut list item 1: no sliders.
PALETTE_SIZE = 6
PALETTE_SAMPLE_PX = 200
KMEANS_SEED = 0


def load_rgb(image_bytes, max_dimension):
    """Decode raw image bytes into an EXIF-corrected, resized, uint8 sRGB array.

    Shared by app.py's main display flow and the cached model-call path, so
    both start from the exact same pixels: same rotation, same resize, same
    array. Never upscales, same as .thumbnail()'s own behavior.

    HEIC/HEIF decoding depends on pillow_heif.register_heif_opener() having
    already run once, which app.py does at import time. This function does
    not call it itself: registration is global to Pillow, not scoped to
    whichever file happens to call Image.open.

    Args:
        image_bytes: raw bytes of an uploaded image file.
        max_dimension: long-edge cap in pixels.

    Returns:
        uint8 sRGB array (H, W, 3).
    """
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((max_dimension, max_dimension))
    return np.array(image)


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


def value_range(gray):
    """The actual darkest and lightest pixel in a grayscale array.

    Not assumed to be 0 and 255. A hazy or low-contrast photo genuinely
    doesn't reach either end, and that's information the rubric prompt
    needs, not a bug to correct for.

    Args:
        gray: 2D uint8 array.

    Returns:
        (min, max) as plain Python ints.
    """
    return int(gray.min()), int(gray.max())


# --------------------------------------------------------------------------
# Color space
#
# There is exactly one sRGB -> Lab path in this project and this is it.
#
# cv2.cvtColor returns a different Lab scale depending on the dtype it is
# handed. Feed it uint8 and you get L scaled to 0-255 with a +128 offset on a
# and b. Feed it float32 in 0-1 and you get true CIE Lab, L 0-100 and a/b
# roughly +-127. Both run. Mixing them gives wrong distances and
# plausible-but-wrong paint matches, which is the S7 trap.
#
# The paint reference values are manufacturer-measured true CIE Lab, so the
# float32 path is not a preference, it is forced by the data. Everything that
# gets compared to a paint value comes through here.
# --------------------------------------------------------------------------


def srgb_to_lab(rgb_u8):
    """uint8 sRGB image (H, W, 3) -> float32 true CIE Lab (H, W, 3)."""
    arr = np.asarray(rgb_u8, dtype=np.uint8)
    return cv2.cvtColor(arr.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_srgb(lab):
    """float32 true CIE Lab (H, W, 3) -> uint8 sRGB (H, W, 3).

    Display only. Never feed the result of this back into a paint match; the
    round trip is lossy and reintroduces the mixed-scale problem by hand.
    """
    rgb01 = cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)
    # Round before the cast, same reason as in posterize: astype truncates.
    return np.round(np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample(rgb, max_px):
    """Shrink so the long edge is at most max_px. Never upscales.

    INTER_AREA, not INTER_LINEAR: when shrinking, INTER_AREA averages the
    pixels it is collapsing, which is what keeps the color statistics honest.
    INTER_LINEAR samples and aliases, which would bias the very histogram
    k-means is about to cluster.
    """
    height, width = rgb.shape[:2]
    scale = max_px / float(max(height, width))
    if scale >= 1.0:
        return rgb
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def extract_palette(rgb, k=PALETTE_SIZE, sample_px=PALETTE_SAMPLE_PX, seed=KMEANS_SEED):
    """Cluster an image's colors and return k swatches, most-used first.

    Three decisions live here:

    1. Cluster in Lab, not RGB, so the clusters correspond to colors a painter
       would treat as one mixture. Equal distances in Lab look equally
       different; equal distances in RGB do not.
    2. Sort by cluster population, not lightness or hue. A painter wants to
       know how much of the canvas each color covers, because that is what
       tells them how much to mix. Sorting by lightness makes a prettier strip
       and a less useful one.
    3. Seed the RNG. k-means picks its starting centroids randomly, so the
       same photo run twice returns different swatches. Measured drift on an
       unseeded run reached 41 Lab units, which is a different color, not a
       rounding wobble. Someone re-uploads their photo, gets a different
       palette, and concludes the app is broken, and they are right.

    Args:
        rgb: uint8 sRGB array (H, W, 3), any size.
        k: number of swatches.
        sample_px: long edge to downsample to before clustering.
        seed: RNG seed. Same seed and same image always gives the same result.

    Returns:
        List of k dicts, ordered by share descending, each with:
            "lab"   tuple of 3 floats, true CIE Lab. Use this for paint matching.
            "rgb"   tuple of 3 ints 0-255, for display.
            "hex"   "#rrggbb" string.
            "share" float 0-1, fraction of sampled pixels in this cluster.
    """
    small = downsample(np.asarray(rgb, dtype=np.uint8), sample_px)
    samples = srgb_to_lab(small).reshape(-1, 3)

    if samples.shape[0] < k:
        raise ValueError(
            "need at least %d pixels to find %d clusters, got %d"
            % (k, k, samples.shape[0])
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # Seed immediately before the call. cv2 seeds through global RNG state, so
    # the reset has to be adjacent to the thing it is protecting.
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k)

    # kind="stable" matters: argsort defaults to an unstable sort, so two
    # clusters with identical pixel counts could swap places between runs even
    # with the RNG seeded. Same class of bug, one layer down.
    order = np.argsort(-counts, kind="stable")

    total = float(labels.size)
    swatches = []
    for index in order:
        lab = centers[index]
        rgb_values = lab_to_srgb(lab.reshape(1, 1, 3)).reshape(3)
        swatches.append(
            {
                "lab": tuple(float(v) for v in lab),
                "rgb": tuple(int(v) for v in rgb_values),
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb_values),
                "share": float(counts[index]) / total,
            }
        )
    return swatches


def dominant_temperature(palette):
    """Share-weighted average Lab b value across a palette's swatches.

    The Lab b axis runs blue (negative) to yellow (positive), close enough
    to painting's warm/cool axis to be useful, and free: it reuses the same
    Lab values the palette already carries rather than a new color model.
    Ignores the a axis (green/red) entirely, so this is a simplification,
    not a full color-temperature measurement.

    Weighted by share rather than a plain mean over the k swatches, so a
    palette dominated by one large cool sky and a few small warm accents
    reports as cool, which is what the canvas actually reads as.

    Args:
        palette: list of dicts from extract_palette, each with "lab" and
            "share".

    Returns:
        Float. Positive leans warm (toward yellow), negative leans cool
        (toward blue).
    """
    return sum(swatch["lab"][2] * swatch["share"] for swatch in palette)


# How many value bands the stages cut the photo into. Three because S5
# validated three as the count that reads as connected massing. Shared by
# the value studies, the line drawing's shape labels, and the two color
# stages, so all of them agree on where a dark stops being a dark.
STAGE_VALUE_LEVELS = 3

# The toned canvas the first three stages sit on: a mid-value neutral, which
# is the rubric's own first instruction, and the app is called Underpainting.
# Not white, and this was measured rather than assumed. With white standing
# for uncovered canvas, stage 2 of the sample photo came out 81% blank,
# because the sky and the dune both land in the middle value band, and the
# pale ground that did get painted was nearly white itself. So white was
# reading as the lightest value in some places and as nothing at all in
# others, in the same picture. A mid-value ground can only mean one of those.
TONED_GROUND = 128


# --------------------------------------------------------------------------
# Line drawing
#
# The first thing a painter puts on a canvas is the drawing, and until S12
# the app never showed it: the filmstrip opened on a value study that
# non-painters could not read, while the rubric had been saying "get the
# drawing down first" all along.
#
# Two line sources are unioned because they fail in opposite directions.
# Region outlines can only draw borders between large areas, so anything
# thin (an earring, a lamp cord) or internal to one area (an eye) is
# invisible to them by construction. A contrast edge detector sees exactly
# those and little else. Both then get the same test neither had on its
# own: is this boundary a real edge, measured as the Lab distance between
# the two sides' own mean colors.
#
# Every constant below is a fraction of the image's own long edge rather
# than a pixel count, for the same reason soften's radius was: a fixed
# pixel count means something different on a 400px image than a 1200px one.
# --------------------------------------------------------------------------

# Median filter applied to the label map before contours are taken. Speckle
# in a label map becomes a wobbling line, so it is cheaper to remove it here
# than to filter the lines it would have produced.
LINE_LABEL_SMOOTHING = 0.02

# Keep the 20 largest shapes overall rather than everything above a fixed
# area fraction. An area floor is tuned to one photo's complexity by
# construction: the fraction that keeps a portrait readable buries an
# interior. A fixed count is the same budget for every photo.
LINE_KEEP_SHAPES = 20

# The test both line sources were missing. A smooth studio backdrop still
# gets cut into palette clusters and every cut becomes a border, so the
# region pass invented wobbling lines across the portrait's background; a
# gradual sky-to-sand transition has no sharp gradient anywhere, so the
# edge pass missed the dune ridge entirely. Bands of a backdrop sit a
# couple of Lab units apart and drop out here, sky against sunlit sand is
# tens and survives. Same ΔE the paint matching runs.
LINE_MIN_DELTA_E = 15.0

# Bilateral rather than Gaussian is the whole point of this step: it
# smooths inside a region while leaving the region's borders sharp, which
# is the opposite of what a Gaussian does to the edges Canny is about to
# look for. The Canny thresholds are multiples of each photo's own median
# gradient, not hand-picked numbers, so nothing here is tuned per photo.
LINE_BILATERAL_FRACTION = 0.012
LINE_BILATERAL_SIGMA = 60
LINE_CANNY_LOW_RATIO = 0.66
LINE_CANNY_HIGH_RATIO = 1.33

# Texture produces many short disconnected fragments; a feature produces a
# few long ones. A 1px line's pixel count is its length, so component size
# is the filter.
#
# This pair is the one number on this page that is a taste call rather than
# a measurement: it decides whether the drawing reads as a loose block-in or
# a contour illustration, and there is no correct answer to measure toward.
# Chosen by rendering both ends of the range on all three test photos and
# picking from the pair, not by tuning until one photo looked right. Raising
# it further does keep cleaning up the interior, but it is also what starts
# eating the portrait's earring, so it is not free in either direction.
LINE_EDGE_MIN_LENGTH = 0.20
LINE_REGION_MIN_LENGTH = 0.10

# Line weight, and how far in from the frame to stop. The border of the
# photo is not a line in the drawing; without the margin every image gets
# a rectangle drawn around it, since the frame is a boundary of every
# region that touches it.
LINE_THICKNESS = 0.002
LINE_EDGE_THICKNESS = 0.0018
LINE_FRAME_MARGIN = 0.004


def nearest_palette_index(rgb, palette):
    """Index of the nearest palette swatch for every pixel, in Lab.

    The one place a pixel gets assigned to a swatch. recolor_to_palette
    paints with the result and shape_labels partitions with it, so a change
    to how "nearest" is decided can't apply to one and miss the other.

    Distance is Lab, the same space the palette itself was clustered in, so
    a pixel lands in the cluster it would have joined had it been part of
    the k-means input. Matched at full resolution, not the 200px sample the
    palette was clustered from: the centroids describe the sample, but every
    full-res pixel still has a well-defined nearest one.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: list of dicts from extract_palette (each needs "lab").

    Returns:
        int array (H, W) of indices into palette.
    """
    lab = srgb_to_lab(rgb)
    height, width = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    centers = np.array([p["lab"] for p in palette], dtype=np.float32)

    # One (N,) distance array per center, not one (N, k, 3) array of diffs:
    # k is small (6) but N is up to a few million pixels, and the (N, k, 3)
    # version peaks at k times the memory for no benefit.
    distances = np.stack(
        [np.linalg.norm(flat - center, axis=1) for center in centers], axis=1
    )
    return np.argmin(distances, axis=1).reshape(height, width)


def value_band_index(gray, n_levels):
    """Which posterize band each pixel falls in, as 0 .. n_levels - 1.

    posterize returns the band's display value (0, 128, 255 at three
    levels); this returns the band number. Same quantization, expressed as
    a label rather than a gray, so it can be combined with another label
    map without the arithmetic depending on what 255 happens to mean.
    """
    step = 255.0 / (n_levels - 1)
    return np.round(posterize(gray, n_levels).astype(np.float32) / step).astype(np.uint8)


def shape_labels(rgb, palette, n_levels=STAGE_VALUE_LEVELS,
                 smoothing_fraction=LINE_LABEL_SMOOTHING):
    """Partition the photo into the shapes a painter would block in.

    Color alone merges a lit wall with the same wall in shadow; value alone
    merges a blue sky with an equally light patch of sand. The product of
    the two separates both, which is why the label is palette index times
    n_levels plus value band rather than either one on its own.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: list of dicts from extract_palette.
        n_levels: number of value bands to cut across the palette clusters.
        smoothing_fraction: median filter size, fraction of the long edge.

    Returns:
        uint8 array (H, W) of labels. Values are indices, not meaningful
        numbers: label 7 is not "between" labels 6 and 8.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    long_edge = max(rgb.shape[:2])

    # medianBlur needs an odd kernel and refuses anything below 3.
    kernel = max(3, int(round(long_edge * smoothing_fraction)) | 1)
    color = cv2.medianBlur(nearest_palette_index(rgb, palette).astype(np.uint8), kernel)
    value = cv2.medianBlur(value_band_index(gray, n_levels), kernel)
    return (color.astype(np.int32) * n_levels + value).astype(np.uint8)


def _clear_frame(mask, long_edge, margin_fraction=LINE_FRAME_MARGIN):
    """Blank a margin all the way around. Modifies and returns mask."""
    margin = max(2, int(round(long_edge * margin_fraction)))
    mask[:margin, :] = mask[-margin:, :] = mask[:, :margin] = mask[:, -margin:] = 0
    return mask


def _drop_short_components(mask, long_edge, min_length_fraction):
    """Keep only connected components at least min_length_fraction long."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_length_fraction * long_edge
    return keep[components].astype(np.uint8)


def _thicken(mask, long_edge, thickness_fraction):
    thickness = max(1, int(round(long_edge * thickness_fraction)))
    return cv2.dilate(mask, np.ones((thickness, thickness), np.uint8))


def region_outlines(rgb, labels, keep_n=LINE_KEEP_SHAPES,
                    min_delta_e=LINE_MIN_DELTA_E,
                    min_length_fraction=LINE_REGION_MIN_LENGTH,
                    thickness_fraction=LINE_THICKNESS):
    """Borders between the biggest shapes, where the two sides really differ.

    The survivors are redrawn into a clean label map and its boundaries are
    taken once, rather than each contour being stroked on its own. Two
    neighboring regions share one border, so stroking per contour draws that
    border twice, slightly offset, and a shape sitting inside another one
    gets outlined on top of its parent's fill.

    Each surviving shape's mean Lab is measured from the photo, not from the
    palette centroid it was labeled with, so the ΔE test compares what is
    actually on the canvas rather than which cluster index the pixels landed in.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        labels: label map from shape_labels, same H and W.
        keep_n: how many of the largest shapes to keep, overall.
        min_delta_e: minimum Lab distance between two shapes for their shared
            border to count as a line.
        min_length_fraction: drop components shorter than this fraction of the
            long edge.
        thickness_fraction: line weight as a fraction of the long edge.

    Returns:
        uint8 array (H, W), 1 where a line is, 0 elsewhere.
    """
    height, width = labels.shape
    long_edge = max(height, width)

    candidates = []
    for level in np.unique(labels):
        mask = (labels == level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidates.append((cv2.contourArea(contour), contour))
    candidates.sort(key=lambda c: -c[0])

    # Every shape gets its own index, not its label, so two separate shapes
    # that happen to share a palette-and-value label still get a line between
    # them if they touch.
    cleaned = np.zeros(labels.shape, dtype=np.int32)
    for index, (_, contour) in enumerate(candidates[:keep_n], start=1):
        simple = cv2.approxPolyDP(contour, 0.002 * cv2.arcLength(contour, True), True)
        cv2.drawContours(cleaned, [simple], -1, index, cv2.FILLED)

    lab = srgb_to_lab(rgb)
    count = int(cleaned.max()) + 1
    means = np.zeros((count, 3), dtype=np.float32)
    for index in range(count):
        pixels = lab[cleaned == index]
        if len(pixels):
            means[index] = pixels.mean(axis=0)

    edge = np.zeros((height, width), dtype=np.uint8)
    for axis in (0, 1):
        a = cleaned[:, :-1] if axis else cleaned[:-1, :]
        b = cleaned[:, 1:] if axis else cleaned[1:, :]
        keep = (a != b) & (np.linalg.norm(means[a] - means[b], axis=-1) >= min_delta_e)
        if axis:
            edge[:, :-1] |= keep.astype(np.uint8)
        else:
            edge[:-1, :] |= keep.astype(np.uint8)

    _clear_frame(edge, long_edge)
    # A boundary that survives the ΔE test in scattered patches is not a line,
    # it is dirt, and it is the same dirt the edge pass produces.
    edge = _drop_short_components(edge, long_edge, min_length_fraction)
    return _thicken(edge, long_edge, thickness_fraction)


def contrast_edges(rgb, min_length_fraction=LINE_EDGE_MIN_LENGTH,
                   thickness_fraction=LINE_EDGE_THICKNESS):
    """Canny edges on a bilateral-filtered copy, long fragments only.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        min_length_fraction: drop components shorter than this fraction of the
            long edge. This is the looseness knob for the whole drawing.
        thickness_fraction: line weight as a fraction of the long edge.

    Returns:
        uint8 array (H, W), 1 where a line is, 0 elsewhere.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    long_edge = max(rgb.shape[:2])

    diameter = max(5, int(round(long_edge * LINE_BILATERAL_FRACTION)) | 1)
    smooth = cv2.bilateralFilter(rgb, diameter, LINE_BILATERAL_SIGMA, LINE_BILATERAL_SIGMA)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)

    median = float(np.median(gray))
    edges = cv2.Canny(
        gray,
        int(max(0, LINE_CANNY_LOW_RATIO * median)),
        int(min(255, LINE_CANNY_HIGH_RATIO * median)),
        L2gradient=True,
    )

    mask = _drop_short_components(edges, long_edge, min_length_fraction)
    _clear_frame(mask, long_edge)
    return _thicken(mask, long_edge, thickness_fraction)


def line_drawing(rgb, palette, edge_min_length=LINE_EDGE_MIN_LENGTH,
                 region_min_length=LINE_REGION_MIN_LENGTH):
    """Stage 1: the drawing, in black on the toned ground.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: the same palette extract_palette returned for this rgb, so
            the shapes outlined are the ones the color stages will fill.
        edge_min_length: contrast-edge fragment filter, fraction of long edge.
        region_min_length: region-outline fragment filter, same units.

    Returns:
        uint8 sRGB array (H, W, 3), TONED_GROUND with black lines.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    lines = region_outlines(
        rgb, shape_labels(rgb, palette), min_length_fraction=region_min_length
    ) | contrast_edges(rgb, min_length_fraction=edge_min_length)

    canvas = np.full(rgb.shape, TONED_GROUND, dtype=np.uint8)
    canvas[lines.astype(bool)] = 0
    return canvas


# --------------------------------------------------------------------------
# Stage generation
#
# Four views of the same source image, in the order a painter builds them:
# the drawing, the two ends of the value range in color, the midtones, then
# the photo. Each is computed straight from the untouched rgb array, never
# from another stage's output. Chaining would compound each stage's own
# approximation into the next one, and the last stage would drift off the
# actual photo instead of being it.
#
# Stages 2 and 3 paint with the same six swatches the palette panel shows,
# so the filmstrip stays tied to the palette rather than being an unrelated
# output that happens to sit near it.
# --------------------------------------------------------------------------

# Labels for build_stages's four outputs, in order. Single source of truth:
# app.py uses these as filmstrip captions and rubric.py uses the same
# strings as the step labels it asks the model for, so the written guide
# and the images point at each other instead of running two numbering
# schemes that can drift apart.
STAGE_CAPTIONS = [
    "1 · Drawing",
    "2 · Darks and lights",
    "3 · Midtones",
    "4 · Full detail",
]


def value_block_in(rgb, n_levels=STAGE_VALUE_LEVELS):
    """The value study, as an RGB array. Grayscale, posterized, on 3 channels.

    No longer one of the four stages as of S12, but still the thing the
    value-studies panel shows and still what the GIF-free part of the page
    is checked against, so it keeps its own function rather than being
    inlined into app.py's two st.image calls.
    """
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(posterize(gray, n_levels), cv2.COLOR_GRAY2RGB)


def recolor_to_palette(rgb, palette):
    """Stage 3: every pixel snapped to its nearest palette color.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: list of dicts from extract_palette (each needs "lab" and "rgb").

    Returns:
        uint8 sRGB array (H, W, 3), containing only the palette's own colors.
    """
    colors = np.array([p["rgb"] for p in palette], dtype=np.uint8)
    return colors[nearest_palette_index(rgb, palette)]


def anchor_masses(rgb, palette, n_levels=STAGE_VALUE_LEVELS):
    """Stage 2: the darkest and lightest bands only, in palette color.

    The rubric's first value instruction is to place both ends of the range
    as actual shapes before any midtone, because everything else is judged
    between those two anchors. This stage is that instruction as a picture:
    the same palette colors stage 3 uses, but only where the photo is in its
    darkest or lightest value band. The middle band is left as bare toned
    ground, so the step from here to stage 3 is exactly the midtones arriving
    and nothing else.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: list of dicts from extract_palette.
        n_levels: value bands to cut. The first and last are the anchors, so
            at the default of 3 exactly one middle band is held back.

    Returns:
        uint8 sRGB array (H, W, 3): palette colors in the anchor bands,
        TONED_GROUND everywhere else.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bands = value_band_index(gray, n_levels)

    painted = recolor_to_palette(rgb, palette)
    canvas = np.full(rgb.shape, TONED_GROUND, dtype=np.uint8)
    anchors = (bands == 0) | (bands == n_levels - 1)
    canvas[anchors] = painted[anchors]
    return canvas


def build_stages(rgb, palette):
    """The four stages: drawing, darks and lights, midtones, full detail.

    The soft-focus stage that sat third until S12 is gone. A Gaussian blur
    was always standing in for "detail not yet resolved," which is not a
    thing a painter does to a canvas, and the slot was better spent on the
    drawing step the rubric had been asking for since S9.

    Args:
        rgb: uint8 sRGB array (H, W, 3).
        palette: the same palette extract_palette returned for this rgb, so
            stages 1 through 3 use the identical clusters shown in the
            swatches panel.

    Returns:
        List of 4 uint8 sRGB arrays (H, W, 3), same size as rgb. The last one
        is rgb itself, unmodified, not a fourth approximation of it.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    return [
        line_drawing(rgb, palette),
        anchor_masses(rgb, palette),
        recolor_to_palette(rgb, palette),
        rgb,
    ]


def cross_fade_frames(stages, fade_frames):
    """Linear cross-fade between each consecutive pair of stages.

    Blended in uint8 RGB, not Lab: an animation frame is a display artifact
    on its way to a GIF palette, not a color measurement, and Lab would add a
    round trip through lab_to_srgb for every intermediate frame with nothing
    visible to show for it.

    Args:
        stages: list of uint8 RGB arrays, all the same shape.
        fade_frames: number of interpolated frames generated strictly
            between each pair (the endpoints are included once each, not
            repeated at both ends of their neighboring fades).

    Returns:
        Frames in playback order: stages[0], fade_frames frames toward
        stages[1], stages[1], fade_frames frames toward stages[2], and so on
        through stages[-1]. Length is len(stages) + (len(stages) - 1) * fade_frames.
    """
    frames = [stages[0]]
    for i in range(len(stages) - 1):
        start = stages[i].astype(np.float32)
        end = stages[i + 1].astype(np.float32)
        for step in range(1, fade_frames + 1):
            t = step / (fade_frames + 1)
            blended = start * (1.0 - t) + end * t
            frames.append(np.round(blended).astype(np.uint8))
        frames.append(stages[i + 1])
    return frames
