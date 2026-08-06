"""The saved step-by-step guide served when the live model call fails.

WHY THIS EXISTS
---------------
The app is on a public URL with no authentication and a small prepaid balance. When the
credits run out, the spend cap trips, or the key is rotated, the last button on the page
becomes a red error box and a visitor sees a broken app instead of a finished one. This is
the insurance policy from the build plan's §7.1: the three other outputs are computed
locally and cannot fail that way, so this is the only part of the page that needs one.

It can only cover the sample photo. There is nothing to pre-generate for a photo the app
has never seen, so an upload that hits a failed call gets an honest error and a pointer at
the sample, not a guide written about someone else's picture.

WHAT THIS TEXT IS
-----------------
The literal output of app.generate_writeup for assets/sample-dead-vlei.jpg, produced by
importing app.py and calling that exact function, not a reimplementation of the API call
and not something written by hand to look like one. Left byte for byte as the model
returned it, punctuation included: the moment it is edited, calling it a saved model
response stops being true, which is the one thing this feature must not do.

app.py labels it on screen as saved rather than live. That note is the point of the
feature, not a disclaimer bolted onto it.

REGENERATING IT
---------------
Whenever the rubric text changes, and therefore whenever app.RUBRIC_VERSION is bumped:
import app in a plain Python shell, call generate_writeup on the sample photo's bytes,
paste the result below, and update the three constants. test_demo_writeup.py fails until
they agree, so a bumped rubric with a stale saved guide is caught by running the checks
rather than by a visitor reading advice the current rubric no longer gives.
"""

import hashlib

# What this text was generated under. app.py compares RUBRIC_VERSION against its own and
# says so on screen if they have drifted apart; test_demo_writeup.py asserts on all three.
RUBRIC_VERSION = "v3"
MODEL = "claude-sonnet-5"
GENERATED_ON = "2026-08-06"

# The sample photo this guide describes. A saved guide about a dead tree served next to a
# different photograph would be worse than no guide at all, so the file it was written
# from is pinned by content, not by filename.
SAMPLE_SHA256 = "de5ec2d7613ffaa7971d59a3150748e375aea97ca6ab5935e6518084133fadc8"

WRITEUP = """\
1 · Drawing
Tone the canvas with a mid-gray wash first and let it dry — this replaces the white ground so both the near-black tree and the pale desert floor can be judged correctly. Then loosely block in the composition: the dead tree standing just left of center, its trunk splitting into a wide web of bare branches reaching up and to the right; the rounded dune rising behind it from lower-left to upper-right; the flat pale sand floor in front, with the low scrubby grass clumps and fallen branch fragments scattered along the base. Get the tree's proportions and branch angles right before anything else — this silhouette carries the whole image.

2 · Darks and lights
Place your two anchors. The darkest dark is the tree's trunk and the shadowed crevices in the bark and hollow — this photo's measured range hits true black, so that dark can go all the way down. The lightest light is not the sky (it's a saturated mid-blue) but the small pale marks on the upper branches and the sun-bleached ground — the range also hits true white, so a few bright spots deserve full white, saved for last as the sharpest accents. Squint and find roughly three masses: dark tree, warm dune, pale ground/sky split.

3 · Midtones
Build the dune's rust-orange gradient and the flat cerulean sky as broad, mostly unified color shapes — the sky especially can stay simple since it's not the subject. The measured temperature reads cool overall (-16.7), because the huge blue sky dominates the average; paint the dune's orange as only relatively warm against that surrounding blue and pale sand, not warm in isolation. Keep the tree's bark color muted and grayish-brown against both. Start deciding edges: the tree's silhouette against the sky should stay crisp, while the dune's base blurring into the pale ground can stay soft.

4 · Full detail
Step back and squint: does the tree's branch structure still read against the dune, is the sky-to-dune-to-ground color logic holding, does the tree feel appropriately cool-neutral against the warmer dune. Then add texture — the gnarled bark grooves, the scattered dry grass tufts, the fallen branch pieces in the foreground — but concentrate that detail on the tree itself, since it's the subject. Let the dune and sky stay comparatively simple. Stop once the thumbnail reads correctly and the sky and dune haven't picked up detail that competes with the tree."""


def matches_sample(image_bytes):
    """True if these are the bytes of the photo this guide was written from.

    Byte identity, not a similarity test: the guide names a dead tree, a dune, and cracked
    ground, so anything other than that exact file makes it wrong rather than approximate.
    """
    return hashlib.sha256(image_bytes).hexdigest() == SAMPLE_SHA256
