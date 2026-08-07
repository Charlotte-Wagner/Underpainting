"""Cutting the model's written guide into one slice per painting stage.

Text in, text out. No Streamlit imports and no image math, the same standard
supplies.py and paints.py hold to, so this is checkable against a handful of
hand-typed strings instead of only through the browser.

WHY THIS IS NOT A ONE-LINER
---------------------------
rubric.OUTPUT_INSTRUCTIONS already asks the model for exactly this shape: four
steps, each headed by one of imaging.STAGE_CAPTIONS's exact strings, in order,
no preamble, no closing summary. So the split is finding four known labels and
cutting between them, which sounds like str.split and is not.

What the prompt asks for and what comes back are different questions. Measured
on real replies rather than assumed: the saved guide in demo_writeup.py heads
each step with the bare label on its own line, and a live call on a second photo
was checked before this function was trusted. Both agreed, but a model is free
to wrap the same label in a markdown header or bold on any future call, and a
splitter written against one observed formatting would then quietly return
nothing. So the label match is done on a stripped line rather than on the raw
one, and every decoration that would still read as the same heading to a human
is stripped before comparing.

The failure mode this file exists to avoid is silent text loss. A splitter that
drops a paragraph when the formatting shifts is worse than the single block it
replaced, because the single block was at least all there. Hence two rules:

  - Nothing is discarded. Text before the first label is real advice the model
    wrote, so it joins the first slice rather than being thrown away, even
    though the prompt asked for no preamble.
  - Failure is total and visible, never partial. If any label is missing or the
    four are out of order, split_guide returns None and the caller falls back
    to showing the whole guide as one block. Returning three good slices and
    one empty one would put a blank panel in front of a visitor and look like
    the model had nothing to say about that stage.
"""

# Decoration a model may wrap a heading in without meaning a different heading.
# Stripped from both ends of a line before it is compared to a label, so
# "## 3 · Midtones", "**3 · Midtones**" and "3 · Midtones:" all match the label
# "3 · Midtones". Ordinary text lines survive this untouched, and a line that
# happens to strip down to a label is a heading by any reading a visitor would
# give it.
HEADING_DECORATION = "#*_ \t:.-"


def _looks_like(line, label):
    """True if this line is that label, whatever the model wrapped it in.

    Compared case-insensitively because the labels carry no meaning in their
    capitalization, and a model that writes "3 · MIDTONES" as a heading has
    still written the heading.
    """
    return line.strip(HEADING_DECORATION).casefold() == label.casefold()


def split_guide(text, labels):
    """Cut a written guide into one slice per label.

    Args:
        text: the model's reply, or any guide written to the same shape. The
            saved fallback in demo_writeup.py is deliberately included in that,
            since it was produced by the real call and goes through this same
            path on screen.
        labels: the stage labels to cut on, in the order they must appear.
            Always imaging.STAGE_CAPTIONS in the app; a parameter rather than
            an import so this file has nothing to say about image math.

    Returns:
        A list of slices, one per label, each the text under that label with
        the label line itself removed (the panel beside it already shows the
        caption, so repeating it would print the heading twice). Or None if
        the labels are not all present in order, which is the caller's signal
        to show the guide unsliced rather than to show it wrong.
    """
    lines = text.splitlines()

    # One pass, tracking which label is expected next, rather than searching
    # for each label independently. Order is part of what is being checked:
    # four labels found in the wrong order is a reply this function cannot
    # honestly cut, not a reply to reorder on the model's behalf.
    found_at = []
    expected = 0
    for index, line in enumerate(lines):
        if expected < len(labels) and _looks_like(line, labels[expected]):
            found_at.append(index)
            expected += 1

    if len(found_at) != len(labels):
        return None

    slices = []
    for position, start in enumerate(found_at):
        end = found_at[position + 1] if position + 1 < len(found_at) else len(lines)
        # start + 1 drops the label line itself. The slice keeps its internal
        # blank lines, since paragraph breaks are what make a 100-word step
        # readable, and only the outer whitespace is trimmed.
        slices.append("\n".join(lines[start + 1:end]).strip())

    # Anything above the first label is text the model wrote and a visitor
    # should see. It belongs to the first step by position, which is also
    # where OUTPUT_INSTRUCTIONS says preamble material like toning the ground
    # should have gone in the first place.
    preamble = "\n".join(lines[:found_at[0]]).strip()
    if preamble:
        slices[0] = f"{preamble}\n\n{slices[0]}".strip()

    return slices
