"""Reference paint colors, and nearest-tube matching.

WHERE THESE NUMBERS COME FROM
-----------------------------
CIE Lab values for Williamsburg Artist Oil Colors, published by Golden Artist
Colors. Measured from 6-mil wet drawdowns (about two sheets of paper thick),
read while wet with a non-contact spectrophotometer.

Source PDF:
https://justpaint.org/wp-content/uploads/2017/06/Munsell-and-CIELAB-Data-for-Williamsburg-Oils_munsell_ordering.pdf
Method: https://justpaint.org/munsell-notations-for-golden-and-williamsburg-paints/

Known limitation, stated because it is real: the published table does not
specify its illuminant or standard observer. This is not calibrated accuracy.

These rows were extracted from the PDF by script and written into this file by
script. They were never retyped by hand, because transcription is the exact
failure mode this sourcing decision exists to avoid.

SELECTION
---------
21 tubes, a deliberately limited palette, the way a paint box is actually set
up. Nobody works from a 100-tube catalog.

  - Where Williamsburg publishes variants, the unprefixed name is used. "SF"
    is the safflower-oil binder; "French" and "Italian" are regional earth
    variants. The plain name is the flagship.
  - Pyrrole Red was in the original palette plan, but Williamsburg publishes
    no measurement for it. Cadmium Red Medium replaces it and fills the same
    gap: high-chroma red at L 39, between Cadmium Red Light (L 51) and
    Quinacridone Magenta (L 19).
  - Graphite Grey was added after measuring. Without it the reference set has
    neutrals at L 97-99 (the whites) and at L 34 and below, and nothing
    between, so every neutral mid-tone in a photo matched whichever chromatic
    tube happened to be least far away. A mid grey was returning Chromium
    Oxide Green at dE 35.5. With Graphite Grey it returns Graphite Grey at
    26.1, and a mid-dark grey improved from Burnt Umber at 27.5 to 11.5.
    Davy's Grey Deep was tested too and never won, so it is not included.

THE MASSTONE CAVEAT, which explains why the distances look large
---------------------------------------------------------------
These are masstones: paint straight from the tube at full strength. Photo
centroids are mostly mid-values and tints. A pale sky lands a long way in
Delta E from every masstone here, because no tube color out of the tube is
pale. That is not a bug and must not be "fixed."

The app does not say what color something is. It says which tube you would
reach for to mix it.

Measured on a real photo, matches ran dE 11.5 to 31.1. A per-item "approximate"
flag at dE 12 would fire on nearly everything and therefore mean nothing, so
there is no per-item threshold. The panel is framed once, as closest tube.

Underpainting is independent and is not affiliated with or endorsed by Golden
Artist Colors or Williamsburg.
"""

import numpy as np

PAINT_REFERENCE = [
    {"name": "Titanium White", "lab": (98.24, -0.99, 2.58), "munsell_hue": "2.7GY"},
    {"name": "Cadmium Lemon", "lab": (89.99, -8.36, 94.94), "munsell_hue": "9.06Y"},
    {"name": "Cadmium Yellow Medium", "lab": (83.08, 16.8, 111.85), "munsell_hue": "2.73Y"},
    {"name": "Cadmium Orange", "lab": (61.2, 55.6, 86.32), "munsell_hue": "1.89YR"},
    {"name": "Yellow Ochre", "lab": (55.5, 18.84, 52.68), "munsell_hue": "9.62YR"},
    {"name": "Cadmium Red Light", "lab": (51.19, 64.23, 70.2), "munsell_hue": "9.29R"},
    {"name": "Raw Sienna", "lab": (44.39, 17.98, 39.66), "munsell_hue": "8.31YR"},
    {"name": "Cadmium Red Medium", "lab": (39.08, 60.78, 48.2), "munsell_hue": "8R"},
    {"name": "Chromium Oxide Green", "lab": (38.97, -19.81, 18.86), "munsell_hue": "9.77GY"},
    {"name": "Cerulean Blue", "lab": (37.28, -13.0, -36.12), "munsell_hue": "2.42PB"},
    {"name": "Graphite Grey", "lab": (33.9, 0.5, 0.82), "munsell_hue": "7.6YR"},
    {"name": "Cobalt Blue", "lab": (26.58, 8.91, -49.14), "munsell_hue": "6.42PB"},
    {"name": "Burnt Sienna", "lab": (25.87, 23.89, 23.2), "munsell_hue": "2.23YR"},
    {"name": "Viridian", "lab": (21.45, -30.2, 1.05), "munsell_hue": "1.51BG"},
    {"name": "Quinacridone Magenta", "lab": (19.37, 31.88, 9.36), "munsell_hue": "6.33R"},
    {"name": "Raw Umber", "lab": (18.68, 0.42, 7.03), "munsell_hue": "6.41Y"},
    {"name": "Burnt Umber", "lab": (18.68, 3.5, 6.17), "munsell_hue": "9.07YR"},
    {"name": "Sap Green", "lab": (17.57, -13.25, 8.38), "munsell_hue": "1.13G"},
    {"name": "Manganese Violet", "lab": (12.4, 31.87, -25.04), "munsell_hue": "8.53P"},
    {"name": "Ivory Black", "lab": (10.83, -0.03, -0.34), "munsell_hue": "5.16PB"},
    {"name": "Ultramarine Blue", "lab": (4.78, 24.88, -37.9), "munsell_hue": "9.01PB"},
]

# Converted once at import, not per request. Shape (21, 3), float32, true CIE
# Lab on the same scale imaging.srgb_to_lab produces.
_PAINT_LAB = np.array([p["lab"] for p in PAINT_REFERENCE], dtype=np.float32)
_PAINT_NAMES = [p["name"] for p in PAINT_REFERENCE]


def nearest_paint(lab):
    """Nearest reference tube to one CIE Lab color, by Delta E 76.

    Delta E 76 is plain Euclidean distance in Lab, which is np.linalg.norm and
    zero new dependencies. CIEDE2000 is more accurate when discriminating
    near-identical samples, but this matches against 21 widely separated tube
    colors, where both metrics pick the same winner nearly every time. Choosing
    the cheaper metric for a stated reason beats pulling ~40MB of scikit-image
    into a free-tier container for one distance function.

    Duplicate matches are allowed on purpose. Two clusters choosing the same
    tube is information: the photo really is dominated by one mixture.

    Args:
        lab: sequence of 3 floats, true CIE Lab. Must come from
            imaging.srgb_to_lab, never from a Lab -> RGB -> Lab round trip.

    Returns:
        (name, delta_e) for the closest tube.
    """
    query = np.asarray(lab, dtype=np.float32).reshape(1, 3)
    distances = np.linalg.norm(_PAINT_LAB - query, axis=1)
    index = int(np.argmin(distances))
    return _PAINT_NAMES[index], float(distances[index])
