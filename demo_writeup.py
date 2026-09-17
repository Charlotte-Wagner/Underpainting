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
GENERATED_ON = "2026-09-17"

# The sample photo this guide describes. A saved guide about a dead tree served next to a
# different photograph would be worse than no guide at all, so the file it was written
# from is pinned by content, not by filename.
SAMPLE_SHA256 = "de5ec2d7613ffaa7971d59a3150748e375aea97ca6ab5935e6518084133fadc8"

WRITEUP = """\
1 · Drawing
Tone your canvas with a thin mid-gray wash and let it dry. Then lightly sketch the dead tree trunk rising just left of center, its branches forking upward and outward against the sky, and the long broken branch lying across the pale sandy foreground at its base. Block the horizon line low, roughly a third up the canvas, where the flat cream-colored basin meets the base of the orange dune. Get the dune's rolling silhouette and the tree's twisting proportions right now, loosely, before any value work begins.

2 · Darks and lights
Your darkest dark is the tree itself: its trunk and branches are near-black, and the measured range confirms this photo reaches true black. Place that first as a shape, not a flat silhouette, find the shadow side of the trunk versus the sunlit side catching warm light. Your lightest light is the deep saturated blue sky, which reads as the brightest large mass here even though it is not white; the true white in this photo is reserved for tiny highlights on the upper branch tips. Squint and confirm three masses: black tree, orange dune, blue sky.

3 · Midtones
Work the orange dune's gradients between the dark tree and light sky: it lightens and grays slightly toward its upper ridge where sun hits it directly, and holds deeper rust in its lower folds. The pale basin floor in front, with its cracked dried mud and scattered green-gray scrub bushes, is a cooler, desaturated midtone; keep it simple since it is not the subject. The measured temperature here leans cool overall (-16.7), so resist over-warming the sand: let the blue sky's coolness and the basin's grayish cast balance the dune's orange. Keep edges of the trunk hard against the sky, softer where the dune meets the pale ground.

4 · Full detail
Step back and squint: does the tree read as the clear subject against the dune and sky. Now render the tree's bark texture, gnarled twists, and the sharp small branch tips, since this is where the eye should land. Add the fine cracked pattern in the foreground mud and a few crisp highlights on the upper branches and dune ridge, saved for last. Let the distant dune stay broad and unified, and the scrub bushes remain soft, unrendered shapes. Stop once the tree holds all the detail and the dune and sky remain simple."""


def matches_sample(image_bytes):
    """True if these are the bytes of the photo this guide was written from.

    Byte identity, not a similarity test: the guide names a dead tree, a dune, and cracked
    ground, so anything other than that exact file makes it wrong rather than approximate.
    """
    return hashlib.sha256(image_bytes).hexdigest() == SAMPLE_SHA256
