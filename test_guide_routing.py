"""Which photo pays: the sample takes the saved guide, an upload calls the model.

Run with:  python test_guide_routing.py

The guide stopped being an optional button and became part of "Let's start!", which is
only affordable because the two photos take different routes. The sample photo -- the
path most visitors take, and the one a stranger clicking a link takes -- is answered
from a file that is already in the repository, and must never reach the API. Anything
somebody uploaded gets the live call, because there is nothing saved that describes it.

That split is a spend property, not a display one, so it is worth a check of its own:
a regression here would not look wrong on screen, it would look like a bill. The rule
asserted below is the strong one -- not "the sample usually uses the saved guide" but
"the sample never calls the model" -- because that is the version that costs money when
it breaks.

Needs no API key, no network and no browser. The model call is replaced with a function
that fails if it is ever reached, which is how "never called" is checked rather than
assumed, and the key reader is stubbed so the live branch is exercised on a machine that
has no secrets file rather than stopping at the missing-key guard in front of it.
"""

import logging

# Same reason as test_api_failures.py: importing app.py outside `streamlit run` logs a
# bare-mode warning per widget. The import is the point -- these are the app's real
# functions, not a copy of their logic.
logging.disable(logging.WARNING)

import app  # noqa: E402
import demo_writeup  # noqa: E402


def rule(title):
    print()
    print(title)
    print("-" * len(title))


class FakeSessionState:
    """st.session_state, which does not exist outside a script run.

    Only the two keys _prepare_guide reads. An attribute bag rather than a dict,
    because the app reads st.session_state.use_sample with a dot.
    """

    def __init__(self, use_sample):
        self.use_sample = use_sample


class ModelWasCalled(AssertionError):
    """Raised in place of the API call, so reaching it fails the check by name."""


def drive(image_bytes, use_sample, existing_guide, allow_call):
    """Run _prepare_guide with everything around it stubbed, and report what happened.

    Returns (stored, called): the guide dicts handed to store_guide, and whether the
    model call was reached. What is being checked is the route taken, and the route is
    visible in exactly those two things.
    """
    stored = []
    called = []
    real = (app.store_guide, app.generate_writeup, app.current_guide,
            app.st.session_state, app.configured_api_key)

    def fake_generate(*args, **kwargs):
        called.append(True)
        if not allow_call:
            raise ModelWasCalled("the model call was reached for the sample photo")
        return "1 · Drawing\nx\n2 · Darks and lights\nx\n3 · Midtones\nx\n4 · Full detail\nx"

    app.store_guide = stored.append
    app.generate_writeup = fake_generate
    app.current_guide = lambda: existing_guide
    app.st.session_state = FakeSessionState(use_sample)
    # Stubbed, not read. _generate_and_store_guide checks for a key before it calls
    # anything, so on a machine with no secrets file the live branch would stop short
    # of the call and rule 3 would pass for the wrong reason -- which is how the first
    # version of this file passed locally and failed the moment it was run against a
    # bare checkout. No real key is involved either way: CLAUDE.md's rule is that no
    # check may depend on one, and stubbing the reader is what keeps that true.
    app.configured_api_key = lambda: "sk-ant-placeholder"
    try:
        app._prepare_guide(image_bytes)
    finally:
        (app.store_guide, app.generate_writeup, app.current_guide,
         app.st.session_state, app.configured_api_key) = real
    return stored, bool(called)


SAMPLE_BYTES = app.SAMPLE_IMAGE_PATH.read_bytes()
NOT_THE_SAMPLE = b"\x89PNG\r\n\x1a\n" + b"not the sample photo"


# --------------------------------------------------------------------------
rule("1. The sample photo is answered from the saved file, with no model call")

stored, called = drive(SAMPLE_BYTES, use_sample=True, existing_guide=None,
                       allow_call=False)
assert not called, "the sample photo reached the model call"
assert len(stored) == 1, f"stored {len(stored)} guides, expected exactly 1"
assert stored[0]["text"] == demo_writeup.WRITEUP, "the sample did not get the saved text"
print(f"  stored the saved guide, {len(stored[0]['text'].split())} words, no call made")
print("PASS: the free path is free")


# --------------------------------------------------------------------------
rule("2. The saved guide does not claim an outage that never happened")

guide = app.saved_guide_for_sample(SAMPLE_BYTES)
print(f"  notice: {guide['notice'][:66]}...")
assert guide["notice_kind"] == "info", "the default guide reported as an error"
# The failure wording belongs to saved_guide_or_error, which is reached only from a
# real failure. Borrowing it here would tell every visitor the API was down.
for claim in ("unavailable", "failed", "outage", "rather than one written just now"):
    assert claim not in guide["notice"], f"the default notice claims {claim!r}"
assert "saved" in guide["notice"], "the default notice does not say the guide is saved"
assert guide["caption"] == app.saved_guide_provenance(), "provenance line went missing"
print("PASS: it says saved, not broken")


# --------------------------------------------------------------------------
rule("3. An uploaded photo does call the model")

stored, called = drive(NOT_THE_SAMPLE, use_sample=False, existing_guide=None,
                       allow_call=True)
assert called, "an upload did not reach the model call"
print(f"  call reached, {len(stored)} guide stored")
print("PASS: the live path is still live")


# --------------------------------------------------------------------------
rule("4. The photo's own bytes decide, not the session flag")

# use_sample says sample, the bytes say otherwise. The hash is the authority, for the
# same reason saved_guide_or_error checks it: the saved guide names a dead tree and a
# dune, so serving it beside anything else describes a photo that is not on the page.
stored, called = drive(NOT_THE_SAMPLE, use_sample=True, existing_guide=None,
                       allow_call=True)
assert called, "a mislabelled photo was served the sample's saved guide"
assert app.saved_guide_for_sample(NOT_THE_SAMPLE) is None, (
    "saved_guide_for_sample answered for a photo it was not written from"
)
print("  a non-sample photo flagged as the sample still went to the live call")
print("PASS: the bytes win")


# --------------------------------------------------------------------------
rule("5. A guide already in hand is not bought twice, but a failure is retried")

usable = {"text": "already written", "notice": None, "notice_kind": None, "caption": None}
stored, called = drive(NOT_THE_SAMPLE, use_sample=False, existing_guide=usable,
                       allow_call=False)
assert not called, "a second press bought a second call"
assert stored == [], "a second press overwrote a guide that was already there"
print("  usable guide in hand -> no call, nothing stored")

# A guide with no text is a failure, not an answer, and is worth another attempt.
failed = {"text": None, "notice": "it broke", "notice_kind": "error", "caption": None}
stored, called = drive(NOT_THE_SAMPLE, use_sample=False, existing_guide=failed,
                       allow_call=True)
assert called, "a failed guide was treated as an answer and never retried"
print("  failed guide in hand  -> retried")
print("PASS: repeats are free, failures get another go")


print()
print("ALL CHECKS PASSED")
