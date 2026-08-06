"""Rule 8 for the demo-mode fallback.

Run with:  python test_demo_writeup.py

Checks the two ways a saved guide goes quietly wrong: the rubric moves and the guide
doesn't, or the sample photo is swapped and the guide keeps describing the old one.
Neither shows up as an error at runtime, which is exactly why they're asserted here.

Whether the fallback actually fires is not checkable from a terminal, because it needs a
failed API call and a browser. That was tested by breaking the key on purpose and clicking
the button; see the S11 entry in debug-log.md.
"""

import logging

# app.py is Streamlit UI, so importing it outside `streamlit run` logs a bare-mode warning
# for every widget on the page. The import itself is the point: RUBRIC_VERSION has to be
# compared against the real constant the app serves under, not a copy of it.
logging.disable(logging.WARNING)

import app  # noqa: E402
import demo_writeup  # noqa: E402
from imaging import STAGE_CAPTIONS  # noqa: E402


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. The saved guide was generated under the rubric and model the app still runs")

print(f"  saved under rubric {demo_writeup.RUBRIC_VERSION}, app serves rubric {app.RUBRIC_VERSION}")
assert demo_writeup.RUBRIC_VERSION == app.RUBRIC_VERSION, (
    "app.RUBRIC_VERSION was bumped without regenerating demo_writeup.WRITEUP: "
    "the saved guide teaches a rubric this app no longer sends"
)
print(f"  saved from {demo_writeup.MODEL}, app calls {app.MODEL_NAME}")
assert demo_writeup.MODEL == app.MODEL_NAME, (
    "the saved guide came from a different model than the live path calls"
)
print("PASS: saved guide and live call agree on rubric version and model")


# --------------------------------------------------------------------------
rule("2. The saved guide is pinned to the sample photo actually committed in assets/")

sample_bytes = app.SAMPLE_IMAGE_PATH.read_bytes()
print(f"  {app.SAMPLE_IMAGE_PATH.name}, {len(sample_bytes)} bytes")
assert demo_writeup.matches_sample(sample_bytes), (
    "the committed sample photo is not the one the saved guide was written from: "
    "regenerate the guide or restore the photo"
)

# The discriminating half. A hash check that says yes to everything would let the saved
# guide be served next to an uploaded photo it says nothing true about.
assert not demo_writeup.matches_sample(b"not the sample photo")
assert not demo_writeup.matches_sample(sample_bytes + b"\x00")
print("PASS: matches the committed photo, rejects other bytes and a one-byte change")


# --------------------------------------------------------------------------
rule("3. The saved guide has the same four labels the live writeup is asked for")

positions = [demo_writeup.WRITEUP.find(caption) for caption in STAGE_CAPTIONS]
print(f"  STAGE_CAPTIONS: {STAGE_CAPTIONS}")
print(f"  found at positions: {positions}")
assert all(p != -1 for p in positions), "a stage label is missing from the saved guide"
assert positions == sorted(positions), "the saved guide's steps are out of build order"
print("PASS: four labels, in build order, matching the filmstrip beside it")


# --------------------------------------------------------------------------
rule("4. The saved guide is about this photo specifically, not generic painting advice")

# The failure this catches is a saved guide that reads like the rubric with the photo
# taken out, which would satisfy every check above. These words are in the guide because
# they're in the picture: a dead tree on cracked ground in front of a dune.
words = demo_writeup.WRITEUP.lower()
subject_terms = ["tree", "dune", "sky"]
print(f"  words: {len(demo_writeup.WRITEUP.split())}, subject terms found: "
      f"{[t for t in subject_terms if t in words]}")
for term in subject_terms:
    assert term in words, f"the saved guide never mentions {term!r}, which is in the photo"
assert len(demo_writeup.WRITEUP.split()) > 200, "the saved guide is too short to be a real reply"
print("PASS: names what's actually in the sample photo")

print()
print("ALL CHECKS PASSED")
