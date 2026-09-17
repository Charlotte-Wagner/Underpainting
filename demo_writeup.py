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
Tone your canvas with a thin mid-gray wash first and let it dry; this becomes the ground you build both directions from. Then lightly draw the dead tree, centered but slightly right of middle, trunk splitting low into a wide fork of bare branches reaching up and out asymmetrically, the widest branch stretching left near the top, a lower broken limb lying flat toward the ground on the right. Block the dune's diagonal line rising left to right behind it, and the pale flat sand floor in front with the low scrubby bushes scattered along the base. Get the tree's proportions and branch angles right before anything else; this composition depends entirely on that silhouette against the sky.

2 · Darks and lights
Your darkest dark is the tree itself: near-black bark in the trunk's shadowed crevices and the undersides of branches. Your lightest light is the sky, but check it: this photo's range runs from true 0 to true 255, so that saturated blue at top is genuinely near pure and the sand highlights on the dune crest genuinely near white. Place both extremes as shapes now. Squint and you should see roughly three masses: black tree silhouette, warm orange-brown dune, and the pale sand foreground. Don't flatten the trunk into one value; it has a lit side and a shadowed side that separate the rounded form from the branch tangle above it.

3 · Midtones
Work the dune's mid-value orange next, graduating slightly darker where it meets the sand floor and slightly lighter near its sunlit crest. The sky's midtone blue sits between your zenith dark-blue and horizon lighter-blue; keep that gradient soft since it recedes. This scene measures cool overall (b axis -16.7), so resist warming the sand or dune more than the photo supports; the orange should read as the warmest note precisely because everything around it, sky and pale foreground, is cooler by comparison. Keep the tree's edge against the sky hard and crisp, since that's the focal contrast, while the dune's edge against sky can stay a touch softer to let it recede.

4 · Full detail
Squint now: does the tree's silhouette still read clearly against the dune, is the dune convincingly behind it, does the cool blue sky and warm dune contrast hold? Only then add detail. Render the bark texture, twisting grain, and small broken branch stubs on the trunk, plus the sharp bright highlight where sun hits the dune's crest, last. Leave the distant left-side scrub and far dune edge simple and soft since they're not the subject. Keep detail concentrated on the tree; if the sand foreground starts competing with it in sharpness, stop."""


def matches_sample(image_bytes):
    """True if these are the bytes of the photo this guide was written from.

    Byte identity, not a similarity test: the guide names a dead tree, a dune, and cracked
    ground, so anything other than that exact file makes it wrong rather than approximate.
    """
    return hashlib.sha256(image_bytes).hexdigest() == SAMPLE_SHA256
