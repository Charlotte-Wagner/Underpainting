"""Rule 8 for guide.split_guide.

Run with:  python test_guide.py

Hand-built strings small enough to read in full, plus the one real model reply
this repository actually has on disk. Asserts, so a wrong answer fails loudly.

The interesting checks here are the failure ones. Rules 4 through 6 are all
variations on "the model formatted it differently than last time", which is the
only way this function can go wrong in production, and rule 7 is the one that
would matter most if it broke: no text is ever dropped on the floor.
"""

from imaging import STAGE_CAPTIONS
from guide import split_guide
import demo_writeup


def rule(title):
    print()
    print(title)
    print("-" * len(title))


LABELS = ["One", "Two", "Three"]


# --------------------------------------------------------------------------
rule("1. Bare labels on their own lines, the shape the saved guide really has")

text = "One\nfirst body\n\nTwo\nsecond body\n\nThree\nthird body"
slices = split_guide(text, LABELS)
print(f"  slices: {slices}")
assert slices == ["first body", "second body", "third body"]
print("PASS: cut on all three labels, label lines themselves removed")


# --------------------------------------------------------------------------
rule("2. The label line is not repeated inside its own slice")

# The panel beside each slice already shows the caption, so a slice that kept
# its heading would print the same words twice on screen.
for label, body in zip(LABELS, slices):
    assert label not in body, f"{label!r} leaked into its own slice"
print(f"  no slice contains its own label: {[label not in body for label, body in zip(LABELS, slices)]}")
print("PASS: headings are consumed by the split, not handed on to the caller")


# --------------------------------------------------------------------------
rule("3. Paragraph breaks inside a step survive; outer whitespace does not")

text = "One\n\nfirst para\n\nsecond para\n\n\nTwo\nb\nThree\nc"
slices = split_guide(text, LABELS)
print(f"  slice 0: {slices[0]!r}")
assert slices[0] == "first para\n\nsecond para", "internal blank line should be kept"
print("PASS: a 100-word step keeps the breaks that make it readable")


# --------------------------------------------------------------------------
rule("4. Markdown headers, bold, and trailing colons all still match")

for decorated in ["## One", "**One**", "One:", "### **One**", "__One__"]:
    text = f"{decorated}\na\nTwo\nb\nThree\nc"
    slices = split_guide(text, LABELS)
    print(f"  {decorated!r:18} -> {slices}")
    assert slices == ["a", "b", "c"], f"{decorated!r} was not recognized as the label"
print("PASS: the same heading in five wrappings cuts identically")


# --------------------------------------------------------------------------
rule("5. A missing label returns None rather than a short list")

text = "One\na\nThree\nc"
result = split_guide(text, LABELS)
print(f"  labels present: One, Three (Two missing) -> {result}")
assert result is None, "a missing label must fail totally, not partially"
print("PASS: two good slices and a hole is not an acceptable answer")


# --------------------------------------------------------------------------
rule("6. Labels out of order return None rather than being reordered")

text = "One\na\nThree\nc\nTwo\nb"
result = split_guide(text, LABELS)
print(f"  order in text: One, Three, Two -> {result}")
assert result is None, "out-of-order labels must fail, not be silently sorted"
print("PASS: build order is the model's to get right, not this function's to repair")


# --------------------------------------------------------------------------
rule("7. Nothing is dropped: a preamble joins step 1 instead of vanishing")

text = "Before you start, tone the ground.\n\nOne\na\nTwo\nb\nThree\nc"
slices = split_guide(text, LABELS)
print(f"  slice 0: {slices[0]!r}")
assert "tone the ground" in slices[0], "preamble text was discarded"
assert slices[1:] == ["b", "c"], "the preamble should not disturb the later slices"

# The real guarantee, stated as a measurement rather than as an intention:
# every non-label line of the input is somewhere in the output.
body_lines = [line for line in text.splitlines() if line.strip() and line not in LABELS]
joined = "\n".join(slices)
missing = [line for line in body_lines if line not in joined]
print(f"  body lines in: {len(body_lines)}, missing from the slices: {missing}")
assert not missing, f"lines lost in the split: {missing}"
print("PASS: no input text is lost, which is the whole reason this returns None on doubt")


# --------------------------------------------------------------------------
rule("8. The saved fallback guide splits, on the app's real STAGE_CAPTIONS")

# demo_writeup.WRITEUP is a byte-for-byte real model reply, so this is the one
# check here running against output nobody wrote to make the test pass. It also
# guards a specific on-screen failure: the fallback was written as one block,
# and it now has to survive the same split the live reply goes through.
slices = split_guide(demo_writeup.WRITEUP, STAGE_CAPTIONS)
assert slices is not None, "the saved guide no longer splits on STAGE_CAPTIONS"
print(f"  slice lengths (chars): {[len(s) for s in slices]}")
assert len(slices) == len(STAGE_CAPTIONS)
assert all(s for s in slices), "a stage came back empty"
for caption, body in zip(STAGE_CAPTIONS, slices):
    assert caption not in body
print("PASS: the real saved reply cuts into four non-empty stages")


# --------------------------------------------------------------------------
rule("9. The split of the saved guide loses none of its words")

rejoined = " ".join(" ".join(slices).split())
original_words = demo_writeup.WRITEUP.split()
label_words = " ".join(STAGE_CAPTIONS).split()
kept = [w for w in original_words if w in rejoined.split() or w in label_words]
print(f"  words in the saved guide: {len(original_words)}, accounted for: {len(kept)}")
assert len(kept) == len(original_words), "the saved guide lost words in the split"
print("PASS: every word of the saved guide is either a label or inside a slice")

print()
print("ALL CHECKS PASSED")
