"""Rule 8 for rubric.build_prompt.

Run with:  python test_rubric.py

Checks prompt assembly, not the rubric's prose quality (that's a by-hand review, not
something to assert on). Asserts, so a wrong answer fails loudly.
"""

from imaging import STAGE_CAPTIONS
from rubric import RUBRIC, build_prompt


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
rule("1. The four stage labels appear in order, exactly as imaging.STAGE_CAPTIONS spells them")

prompt = build_prompt((12, 244), -8.3)
positions = [prompt.find(f'"{caption}"') for caption in STAGE_CAPTIONS]
print(f"  STAGE_CAPTIONS: {STAGE_CAPTIONS}")
print(f"  found at positions: {positions}")
assert all(p != -1 for p in positions), "a stage caption is missing from the prompt verbatim"
assert positions == sorted(positions), "stage captions are not in build order"
print("PASS: all four labels present, in build order, matching the filmstrip exactly")


# --------------------------------------------------------------------------
rule("2. Raw numbers appear, no invented warm/neutral/cool or compressed/full bucket")

prompt = build_prompt((40, 210), 15.7)
print(f"  value_range=(40, 210) -> {'40' in prompt and '210' in prompt}")
assert "darkest pixel 40" in prompt and "lightest pixel 210" in prompt
print(f"  temperature=15.7 -> {'15.7' in prompt}")
assert "15.7" in prompt

# The only axis words allowed are the ones explaining what the number means
# (toward yellow / toward blue), never a category label standing in for a
# threshold that was never measured.
BANNED = ["this photo is warm", "this photo is cool", "this photo is neutral",
          "compressed range", "full range"]
lowered = prompt.lower()
for phrase in BANNED:
    assert phrase not in lowered, f"found an invented category: {phrase!r}"
print("PASS: numbers are present verbatim, no invented bucket standing in for them")


# --------------------------------------------------------------------------
rule("3. Negative temperature formats correctly (sign is meaningful, not noise)")

prompt = build_prompt((0, 255), -22.456)
print(f"  -22.456 -> looking for '-22.5' (one decimal place)")
assert "-22.5" in prompt, "temperature should round to one decimal, sign preserved"
print("PASS: negative sign and rounding both survive into the prompt")


# --------------------------------------------------------------------------
rule("4. The rubric's own text is included whole, not summarized or truncated")

prompt = build_prompt((0, 255), 0.0)
assert RUBRIC.strip() in prompt, "the full rubric text must appear verbatim in the prompt"
print(f"  rubric length: {len(RUBRIC)} chars, fully present in a {len(prompt)} char prompt")
print("PASS: rubric text is not summarized or altered on its way into the prompt")

print()
print("ALL CHECKS PASSED")
