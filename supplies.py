"""What to paint with, keyed on what you are painting on.

Reference data, the same standard as paints.py: no Streamlit imports, no logic
beyond a lookup, so app.py owns the rendering and this file owns the content.

Written by hand from studio practice rather than generated, for the same reason
rubric.py is. The two are the only prose in the app addressed to a painter, and
the point of the project is that a painter wrote them.

Scope note: this is the one part of the S12 stage redesign that was not in the
session brief. It came out of a flow outline mid-session and was added on
request, with a skip control, because a person actually standing in an art shop
needs it and a visitor evaluating the app does not. If it turns out to be in the
way, the whole feature is this file plus one block in app.py.

The surfaces are ordered by how likely a beginner is to be holding one, not
alphabetically. The paint recommendations are deliberately narrow: a beginner
asking "what do I paint on" is not helped by being told everything works.
"""

# Each entry: what to paint with, what else to have on the table, and an
# optional caveat shown under both.
#
# "extras" names the solvent explicitly rather than saying "and a cup of water"
# for everything. Water does nothing for oil paint, and a beginner who sets up
# an oil palette with a water jar finds that out the slow way.
SURFACES = {
    "Stretched canvas or canvas board": {
        "paint": (
            "Acrylic or oil. Both were made for this surface, and anything sold "
            "pre-primed is ready to use as it comes."
        ),
        "extras": (
            "Brushes, a palette, and a rag. Add a jar of water for acrylic, or "
            "odorless mineral spirits and a jar for oil."
        ),
    },
    "Wood panel or primed hardboard": {
        "paint": (
            "Acrylic or oil. A rigid panel takes thick paint, scraping, and "
            "sanding better than stretched canvas does."
        ),
        "extras": (
            "Brushes, a palette, and a rag. Add a jar of water for acrylic, or "
            "odorless mineral spirits and a jar for oil."
        ),
        "caveat": (
            "If the wood is raw rather than primed, seal and prime it first. "
            "Bare wood drinks the paint and stains, and the color you mixed is "
            "not the color that ends up on the panel."
        ),
    },
    "Canvas paper pad": {
        "paint": (
            "Acrylic. Canvas paper will take oil for a study, but it is not "
            "archival with it."
        ),
        "extras": "Brushes, a palette, a jar of water, and a rag.",
        "caveat": (
            "This is a practice surface. Good for repeating the same study "
            "three times cheaply, not for something you want to keep."
        ),
    },
    "Watercolor paper": {
        "paint": (
            "Watercolor or gouache. Cold press, 140lb or heavier, and tape or "
            "stretch it down first or it will buckle as it dries."
        ),
        "extras": "Brushes, a palette, a jar of water, and a towel for lifting.",
        "caveat": (
            "The tube names further down this page are oil colors, matched to "
            "your photo by color alone. On this surface read them as names for "
            "the hues to mix, not as products to buy."
        ),
    },
    "Mixed media paper": {
        "paint": (
            "Acrylic, gouache, or light watercolor. It handles wet work better "
            "than drawing paper and worse than watercolor paper, so keep the "
            "washes light."
        ),
        "extras": "Brushes, a palette, a jar of water, and a rag.",
        "caveat": (
            "The tube names further down this page are oil colors, matched to "
            "your photo by color alone. On this surface read them as names for "
            "the hues to mix, not as products to buy."
        ),
    },
}

# Named rather than left implicit in the list above. These are the surfaces a
# beginner reaches for because they are already in the house, and both fail in
# a way that looks like bad painting rather than like a bad surface.
AVOID = (
    "Two ways this goes wrong before you start: thin paper, meaning printer or "
    "sketchbook paper, cockles and tears under any wet paint, and raw unsealed "
    "wood absorbs it unevenly. Neither one looks like a surface problem while "
    "it is happening. It looks like you cannot control the paint."
)
