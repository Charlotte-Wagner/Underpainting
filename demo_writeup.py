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
RUBRIC_VERSION = "v2"
MODEL = "claude-sonnet-5"
GENERATED_ON = "2026-08-05"

# The sample photo this guide describes. A saved guide about a dead tree served next to a
# different photograph would be worse than no guide at all, so the file it was written
# from is pinned by content, not by filename.
SAMPLE_SHA256 = "de5ec2d7613ffaa7971d59a3150748e375aea97ca6ab5935e6518084133fadc8"

WRITEUP = """\
1 · Values
Tone your canvas a mid-value gray-tan before anything else, so the pale desert floor and deep blue sky both have room to be judged against it. Rough in the dead tree's silhouette, its twisted trunk and reaching bare branches centered in the frame, plus the dune's slope line and the horizon of pale cracked ground. This photo's measured range runs from true 0 to true 255, so you're allowed a genuine black in the tree's deepest crevices and trunk shadow, and a genuine white in the brightest sunlit sand of the dune crest. Place those two anchors first, then midtone the sky's saturated blue and the dune's mid-orange between them.

2 · Color masses
Block in three color masses across the whole canvas at once: the flat saturated blue sky filling the top two-thirds, the warm burnt-orange dune with its darker striated shadow grooves running down the slope, and the pale bone-white cracked earth in the foreground with dull sage-green grass clumps scattered near the tree's base. The measured temperature here is cool overall (-16.7), which may surprise you since the orange dune looks warm at a glance — keep that orange only warm relative to the blue sky and pale ground beside it, don't push it toward true warm or the sky will feel wrong. The tree itself stays a neutral grayish-brown, not warm.

3 · Soft focus
Squint and check: does the tree's silhouette still read against the dune the way it does in the photo, branches clearly separated from the sky, trunk clearly separated from the orange behind it? Decide edges now — the tree against the sky should stay crisp since that's your subject, but the horizon where pale ground meets dune base can stay soft, and the dune's far edge against the sky can soften too, since neither competes with the tree. Keep the grass clumps and ground cracks loose and unresolved. Lock in that the dune reads as one warm mass and the sky as one flat cool mass before moving on.

4 · Full detail
Step back and confirm the silhouette of the tree still reads correctly at a distance before rendering its bark texture, the gnarled knots, and the pale broken branch stubs. Add the fine cracks and mottled tone variation in the foreground earth, and the individual dry blades in the grass clumps near the trunk and fallen branch. Keep the dune simple — a few directional streaks of darker orange for the wind-carved grooves is enough, it is not the subject. Stop once the tree's silhouette and texture hold up at thumbnail size and the dune and sky remain comparatively simple beside it."""


def matches_sample(image_bytes):
    """True if these are the bytes of the photo this guide was written from.

    Byte identity, not a similarity test: the guide names a dead tree, a dune, and cracked
    ground, so anything other than that exact file makes it wrong rather than approximate.
    """
    return hashlib.sha256(image_bytes).hexdigest() == SAMPLE_SHA256
