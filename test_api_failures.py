"""Rule 8 for the model call: every way it fails ends in a notice, not a traceback.

Run with:  python test_api_failures.py

The guide is the optional part of this app, so a failed call should cost a visitor a
line of text and nothing else. Five failures used to escape every handler and put a
Python traceback on the page instead, three of them the plain absence of an API key,
which is the case demo mode exists for and the case anyone deliberately breaking the
secret to watch the fallback would hit first.

Needs no API key, no network and no browser. Nothing here calls the API: the call is
replaced with a function that fails in a named way, which is the only way to test a
handler for an outage without waiting for one.
"""

import logging

# Same reason as test_demo_writeup.py: importing app.py outside `streamlit run` logs a
# bare-mode warning per widget. The import is the point, because these are the app's
# real handlers rather than a copy of their logic.
logging.disable(logging.WARNING)

import anthropic  # noqa: E402
from streamlit.errors import StreamlitSecretNotFoundError  # noqa: E402

import app  # noqa: E402


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# The SDK's errors want a request and a response to have come from somewhere. Stubs
# rather than real httpx objects, because httpx is a dependency of anthropic and not
# of this app: a check that imported it would be resting on a transitive pin that
# requirements.txt does not control.
REQUEST = object()


class FakeResponse:
    """Enough of a response for the SDK's error classes to read a status off."""

    headers = {}
    text = ""
    request = REQUEST

    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return {}


class FakeSecrets:
    """st.secrets, standing in for the three ways this app can have no key.

    A dict is not enough: the absence that matters most is a missing secrets file,
    and that raises rather than returning nothing.
    """

    def __init__(self, mapping=None, raises=None):
        self.mapping = mapping or {}
        self.raises = raises

    def __getitem__(self, key):
        if self.raises is not None:
            raise self.raises
        return self.mapping[key]


def with_secrets(secrets, call):
    """Run call() with app's view of st.secrets replaced, then put it back.

    Restored in a finally because app.st is the streamlit module itself, shared with
    everything else in the process.
    """
    real = app.st.secrets
    app.st.secrets = secrets
    try:
        return call()
    finally:
        app.st.secrets = real


def route(failure, image_bytes, secrets=None):
    """Send one failure through the real handler and return the guide it stored.

    store_guide writes to st.session_state, which does not exist outside a script
    run, so it is captured here instead. What is being checked is which guide the
    handler chooses, and that is the argument it passes.
    """
    stored = []
    real_store, real_generate = app.store_guide, app.generate_writeup
    app.store_guide = stored.append
    app.generate_writeup = failure
    try:
        with_secrets(
            secrets or FakeSecrets({"ANTHROPIC_API_KEY": "sk-ant-placeholder"}),
            lambda: app._generate_and_store_guide(image_bytes),
        )
    finally:
        app.store_guide = real_store
        app.generate_writeup = real_generate
    assert len(stored) == 1, f"handler stored {len(stored)} guides, expected exactly 1"
    return stored[0]


def raises(exc):
    def fail(*args, **kwargs):
        raise exc
    return fail


SAMPLE_BYTES = app.SAMPLE_IMAGE_PATH.read_bytes()
NOT_THE_SAMPLE = b"\x89PNG\r\n\x1a\n" + b"not an image"


# --------------------------------------------------------------------------
rule("1. No usable API key reads as no key, however it is missing")

cases = [
    ("no secrets file at all", FakeSecrets(raises=StreamlitSecretNotFoundError("none"))),
    ("secrets file without the key", FakeSecrets({"OTHER_KEY": "x"})),
    ("the key present but empty", FakeSecrets({"ANTHROPIC_API_KEY": ""})),
    ("the key present but blank", FakeSecrets({"ANTHROPIC_API_KEY": "   "})),
]
for name, secrets in cases:
    got = with_secrets(secrets, app.configured_api_key)
    print(f"  {name:32} -> {got!r}")
    assert got is None, f"{name} did not read as an absent key"

# The discriminating half. A reader that answered None to everything would send a
# healthy app into demo mode forever, which is the failure this check would otherwise
# invite. Not printed: it stands in for a real secret.
present = with_secrets(FakeSecrets({"ANTHROPIC_API_KEY": "sk-ant-placeholder"}),
                       app.configured_api_key)
assert present == "sk-ant-placeholder", "a configured key did not survive the reader"
print("  a configured key                 -> passed through unchanged")
print("PASS: four absences read as None, a real value is not mistaken for one")


# --------------------------------------------------------------------------
rule("2. A missing key never reaches the API at all")

# Not just "does not crash". The call is what costs money and what cannot succeed
# without a key, so the guard has to sit in front of it rather than catch it.
def must_not_run(*args, **kwargs):
    raise AssertionError("the API was called with no key configured")


guide = route(must_not_run, SAMPLE_BYTES,
              secrets=FakeSecrets(raises=StreamlitSecretNotFoundError("none")))
print(f"  notice: {guide['notice'][:72]}...")
assert guide["text"], "no saved guide was served for the sample photo"
assert "no API key is configured" in guide["notice"]
print("PASS: refused before the call, and the visitor still gets the saved guide")


# --------------------------------------------------------------------------
rule("3. Every failure the call can produce ends in a guide, not an exception")

failures = [
    ("timeout", anthropic.APITimeoutError(request=REQUEST)),
    ("connection refused", anthropic.APIConnectionError(request=REQUEST)),
    ("401 rejected key", anthropic.AuthenticationError(
        "bad key", response=FakeResponse(401), body=None)),
    ("429 rate limited", anthropic.RateLimitError(
        "slow down", response=FakeResponse(429), body=None)),
    ("529 overloaded", anthropic.InternalServerError(
        "overloaded", response=FakeResponse(529), body=None)),
    ("400 bad request", anthropic.BadRequestError(
        "bad request", response=FakeResponse(400), body=None)),
    # The two that escaped: neither is an APIStatusError or an APIConnectionError.
    ("unreadable 200 body", anthropic.APIResponseValidationError(
        response=FakeResponse(200), body=None)),
    ("reply with no text", app.EmptyModelReply("the reply carried no text")),
]
for name, exc in failures:
    guide = route(raises(exc), SAMPLE_BYTES)
    reason = guide["notice"].split("(", 1)[1].split(")", 1)[0]
    print(f"  {name:22} -> {reason}")
    assert guide["text"], f"{name} lost the saved guide"
    assert guide["notice_kind"] == "info", f"{name} did not report as an outage"
print("PASS: eight failures, eight notices, no traceback")


# --------------------------------------------------------------------------
rule("4. An uploaded photo gets an honest error rather than the sample's guide")

# The saved guide describes the sample photo, so serving it beside anything else
# would be a guide for a picture that is not on the page. Failing the call must not
# be a way around that.
guide = route(raises(anthropic.APITimeoutError(request=REQUEST)), NOT_THE_SAMPLE)
print(f"  text: {guide['text']!r}, kind: {guide['notice_kind']!r}")
assert guide["text"] is None, "the sample's saved guide was served beside another photo"
assert guide["notice_kind"] == "error", "a failure with no fallback was reported as info"
print("PASS: no saved guide for an unseen photo, and it says so as an error")


# --------------------------------------------------------------------------
rule("5. A healthy call is still stored as a live guide with no notice")

# The check that keeps the four above from passing on a broken app: if everything
# routed to demo mode, the feature would be gone and only this rule would notice.
guide = route(lambda *a, **k: "1 - Drawing\nTone the canvas.", SAMPLE_BYTES)
print(f"  text: {guide['text'][:28]!r}, notice: {guide['notice']!r}")
assert guide["text"].startswith("1 - Drawing"), "a live reply was not stored as written"
assert guide["notice"] is None and guide["caption"] is None, (
    "a healthy call was labelled as a fallback"
)
print("PASS: the live path is untouched by the handlers around it")

print()
print("ALL CHECKS PASSED")
