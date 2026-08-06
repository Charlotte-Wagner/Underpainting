import base64
import time
from io import BytesIO
from pathlib import Path

import anthropic
import cv2
import numpy as np
import pillow_heif
import streamlit as st
from PIL import Image

import demo_writeup
import rubric
from gif import encode_gif, frame_durations
from imaging import (
    STAGE_CAPTIONS,
    build_stages,
    cross_fade_frames,
    dominant_temperature,
    downsample,
    extract_palette,
    load_rgb,
    posterize,
    value_range,
)
from paints import nearest_paint

pillow_heif.register_heif_opener()

st.title("Underpainting")
st.write("Upload a photo and see how a painter would block it in.")

MAX_DIMENSION = 1200

# Lets a stranger with no photo of their own see all four outputs on the
# first try. Chosen for full black-to-white range in one frame and strong
# warm/cool contrast, so it exercises the value study, the palette, and
# the temperature stat, not just the filmstrip.
#
# It's also the easiest thing for a test script to click, though not the
# only one: the OS file-picker dialog genuinely can't be driven, but the
# file input behind it can be populated directly with a DataTransfer, and
# Streamlit treats that as a real upload. Four sessions' notes said the
# upload path was untestable; that was true of the dialog and false of
# the input, and the difference went unchecked until it hid a real bug.
#
# The attribution below is a license condition, not decoration. CC BY-SA
# 4.0 section 3(a)(1) wants the creator named, a link to the source, and
# an indication that the material was modified; the file here is a
# downscaled rendition, and everything the app derives from it on screen
# is modified further. Don't trim this to fit a layout.
SAMPLE_IMAGE_PATH = Path(__file__).parent / "assets" / "sample-dead-vlei.jpg"
SAMPLE_CREDIT = "Dead Vlei, Namibia · Diego Delso, delso.photo, CC BY-SA 4.0"
SAMPLE_SOURCE_URL = (
    "https://commons.wikimedia.org/wiki/"
    "File:Dead_Vlei,_Sossusvlei,_Namibia,_2018-08-06,_DD_086.jpg"
)
SAMPLE_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
SAMPLE_NOTICE = (
    f"Sample photo by Diego Delso, [delso.photo](https://delso.photo), "
    f"[source]({SAMPLE_SOURCE_URL}), licensed "
    f"[CC BY-SA 4.0]({SAMPLE_LICENSE_URL}). Resized from the original, and "
    "modified further by the studies, stages, and animation above, which are "
    "themselves CC BY-SA 4.0."
)

# Hardcoded on purpose. Cut list item 1: no tunable UI, even if on schedule.
# A curated pair of studies is a better demo than a slider that lets a visitor
# find the ugliest output the app can produce.
COARSE_LEVELS = 3
FINE_LEVELS = 5

# The GIF doesn't need filmstrip resolution, and frame count times frame size
# is what makes the file big on a phone on cell data. The filmstrip stays at
# full resolution; only the animation gets downsampled. Measured on the test
# photo: 480px/6 fades was 1.1MB; 300px/4 fades is 368KB with no visible loss
# to the build-order read. See debug-log.md, diamond-4.
GIF_MAX_PX = 300
FADE_FRAMES = 4

# Set by the S9 A/B, not left as a default. Both models were run on both test
# photos, twice each. Both produced writeups specific to the actual photo, and
# the two photos produced writeups with almost no shared content, which is the
# whole verification bar for this session. Opus applied two rubric points more
# consistently (temperature as relative rather than absolute, and saving the
# brightest accent for last) and cost 1.8x: $0.048 a writeup against $0.026, so
# a $5 balance buys ~104 writeups instead of ~190. On a public URL with no
# authentication the balance is a safety net, and the gap was not worth halving
# it. Full comparison in debug-log.md, diamond-5.
MODEL_NAME = "claude-sonnet-5"

# Measured, not estimated. At 800 the first A/B run truncated all four writeups
# mid-sentence: the build plan's ~550-token estimate for a 400-word reply is low
# once markdown formatting and the four step labels are counted, and Opus in
# particular spends tokens on punctuation the estimate didn't anticipate. 1500 is
# roughly double the longest complete reply measured, so the ceiling is headroom
# rather than the thing deciding the length. The word count is controlled by the
# prompt's own limit, not by cutting the reply off.
API_MAX_TOKENS = 1500

# Bump this by hand whenever rubric.RUBRIC's text changes. It's part of the
# cache key below specifically so an edited rubric produces a fresh writeup
# instead of silently returning a stale cached one for the same photo.
RUBRIC_VERSION = "v2"


@st.cache_data(show_spinner=False)
def generate_writeup(image_bytes, rubric_version, model):
    """Photo + measured stats + rubric -> written step-by-step guide.

    Cached on (image_bytes, rubric_version, model), three simple hashable
    values, rather than accepting an already-computed rgb array or palette
    as an argument: recomputing them here costs about 0.2s (measured in S6
    and S8) and avoids the alternative of making Streamlit hash a numpy
    array or a list of dicts, which is slower and easier to get subtly
    wrong. model is part of the key, not just a constant, so the same photo
    can be compared across models without one call's cache entry hiding the
    other's (see the S9 A/B in debug-log.md).

    Exceptions are allowed to propagate out of this function rather than
    being caught here: st.cache_data never caches a raised exception, so a
    failed call is retried on the next click instead of being cached as a
    permanent failure.

    Args:
        image_bytes: raw bytes of the uploaded file, unresized.
        rubric_version: hand-bumped string, see RUBRIC_VERSION above.
        model: Anthropic model name to call.

    Returns:
        The model's written step-by-step guide as plain text.
    """
    rgb = load_rgb(image_bytes, MAX_DIMENSION)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    palette = extract_palette(rgb)
    prompt_text = rubric.build_prompt(value_range(gray), dominant_temperature(palette))

    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    photo_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=API_MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": photo_b64,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        }],
    )
    return next(block.text for block in response.content if block.type == "text")


def show_saved_guide_or_error(image_bytes, short_reason, error_message):
    """Demo mode: the saved guide when the live call fails, or an honest error.

    Only reachable from a failed call, never from a healthy one, so this cannot quietly
    replace the feature it is insuring. Which of the two a visitor gets is decided by the
    photo itself, hashed against the file the saved guide was written from, rather than by
    which button was clicked: the guide names a dead tree, a dune, and cracked ground, so
    serving it beside anything else would describe a photo that isn't on the page. An
    upload therefore gets the S4 error message unchanged, plus a pointer at the one photo
    that does have something saved.

    The on-screen note is the feature, not a disclaimer bolted onto it. Passing a saved
    file off as a live model response is exactly the kind of untrue claim CLAUDE.md rules
    out, and "here is what the model wrote, saved earlier" is a better answer than
    pretending an API never fails. The note also names why the live call failed, so a real
    outage during development still looks like an outage instead of hiding behind a smooth
    fallback.

    Args:
        image_bytes: the bytes currently being displayed, sample or upload.
        short_reason: visitor-facing phrase for why the live call didn't happen.
        error_message: the operator-facing message, kept from S4, for the no-fallback case.
    """
    if not demo_writeup.matches_sample(image_bytes):
        st.error(error_message)
        st.caption(
            "There's nothing saved for a photo the app hasn't seen before. The sample "
            "photo above has a saved guide that works without the API, if you want to "
            "see what this output looks like."
        )
        return

    st.info(
        f"The live model call is unavailable right now ({short_reason}), so this is a "
        "saved guide for the sample photo rather than one written just now. Everything "
        "else on this page was still computed live from the photo."
    )
    st.markdown(demo_writeup.WRITEUP)

    # Same staleness problem the cache key guards against, one layer up: a guide written
    # under one rubric and served under a later one is a guide for advice the app no
    # longer gives. Still shown, because a stale guide beats a red error box, but never
    # shown as if it were current. test_demo_writeup.py fails on the same mismatch, so the
    # usual way to find out is by running the checks, not by a visitor reading it.
    provenance = (
        f"Written by {demo_writeup.MODEL} on {demo_writeup.GENERATED_ON} from this same "
        f"photo, using the same rubric and the same measured stats the live call sends."
    )
    if demo_writeup.RUBRIC_VERSION != RUBRIC_VERSION:
        provenance += (
            f" Saved under rubric {demo_writeup.RUBRIC_VERSION}; the app now runs rubric "
            f"{RUBRIC_VERSION}, so parts of it may not match the current rubric."
        )
    st.caption(provenance)


if "use_sample" not in st.session_state:
    st.session_state.use_sample = False


def _clear_sample():
    """Touching the uploader is an explicit choice to stop using the sample.

    This has to be a callback rather than a plain `if uploaded_file is not
    None` check after the widget. st.file_uploader keeps returning the same
    file on every rerun until it's cleared, so a plain check can't tell "the
    visitor just uploaded this" from "that file has been sitting there since
    three clicks ago". Reading it as the former made the sample button dead
    for anyone who had already uploaded something. Measured, not assumed:
    clicking the sample button with a file loaded left the page on the
    uploaded photo with no feedback at all.
    """
    st.session_state.use_sample = False


uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "heic", "heif"],
    on_change=_clear_sample,
)

st.caption("No photo handy?")
if st.button("Try a sample photo"):
    st.session_state.use_sample = True

# Whichever the visitor chose most recently wins, which is why the sample is
# checked first: the flag is only ever set by a click on the button, and only
# ever cleared by the uploader's own on_change.
if st.session_state.use_sample:
    image_bytes = SAMPLE_IMAGE_PATH.read_bytes()
    source_caption = f"Sample photo · {SAMPLE_CREDIT}"
elif uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    source_caption = "Original"
else:
    image_bytes = None

if image_bytes is not None:
    start = time.time()
    try:
        with st.spinner("Reading your photo..."):
            rgb_array = load_rgb(image_bytes, MAX_DIMENSION)
    except Exception:
        st.error("Couldn't read that file as an image. Try a JPEG, PNG, or HEIC photo.")
    else:
        with st.spinner("Building the value study and palette..."):
            gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

            coarse_study = posterize(gray_array, COARSE_LEVELS)
            fine_study = posterize(gray_array, FINE_LEVELS)
            palette = extract_palette(rgb_array)

        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_array, caption=source_caption)
        with col2:
            st.image(gray_array, caption="Grayscale")

        # Pixel dimensions are noise to a visitor; processing time is a real
        # signal to a technical one ("this ran in a fraction of a second"),
        # so it stays and the dimensions go.
        #
        # Reserved here and filled in after the GIF, because the honest number
        # is the one the visitor actually waited through. Timing only as far
        # as the palette and printing that next to the first image reported
        # 0.12s of a 0.25s pipeline on the sample photo: a precise-looking
        # number that was wrong by half, which is worse than no number.
        timing_caption = st.empty()

        st.markdown("**Value studies**")

        col3, col4 = st.columns(2)
        with col3:
            st.image(coarse_study, caption=f"{COARSE_LEVELS}-value study")
        with col4:
            st.image(fine_study, caption=f"{FINE_LEVELS}-value study")

        st.markdown("**Palette and closest tube**")
        st.caption(
            "Clustered in Lab so the groupings match colors a painter would mix "
            "as one, ordered by how much of the canvas each covers. Each swatch "
            "shows the photo's own color, with the tube you would reach for to "
            "mix it."
        )

        for column, swatch in zip(st.columns(len(palette)), palette):
            # The swatch is the photo's own centroid, never a render of the
            # paint measurement. Showing the paint's color would quietly claim
            # the match is exact.
            paint_name, delta_e = nearest_paint(swatch["lab"])
            with column:
                st.image(np.full((90, 200, 3), swatch["rgb"], dtype=np.uint8))
                st.markdown(f"**{paint_name}**")
                st.caption(
                    f"{swatch['share'] * 100:.1f}% of canvas\n\n"
                    f"{swatch['hex']} · ΔE {delta_e:.0f}"
                )

        with st.expander("How the tube names are chosen"):
            st.markdown(
                "Tube suggestions are approximate. Underpainting compares dominant "
                "sRGB photo colors with measured wet-paint masstones from a limited "
                "reference palette. Camera processing, lighting, transparency, ground "
                "color, mixtures, and drying shifts are not modeled. Use these names "
                "as starting points, not mixing formulas.\n\n"
                "The reference values are masstones, paint straight from the tube at "
                "full strength, while most of a photo is mid-values and tints. That is "
                "why the ΔE numbers are large even when the tube names are right: the "
                "question being answered is which tube to reach for, not what the color "
                "is.\n\n"
                "Reference CIE Lab measurements: [Williamsburg Artist Oil Colors]"
                "(https://justpaint.org/wp-content/uploads/2017/06/"
                "Munsell-and-CIELAB-Data-for-Williamsburg-Oils_munsell_ordering.pdf), "
                "measured from 6-mil wet drawdowns with a non-contact "
                "spectrophotometer. The published table does not specify its "
                "illuminant/observer setting. Underpainting is independent and is not "
                "affiliated with or endorsed by Golden Artist Colors or Williamsburg."
            )

        st.markdown("**Build order**")
        st.caption(
            "The same photo, computed four independent ways: values only, color "
            "reduced to the palette above, detail softened, and untouched. Reading "
            "left to right is the order you would actually build the painting."
        )

        with st.spinner("Building the four stages..."):
            stages = build_stages(rgb_array, palette)
        for column, stage, caption in zip(st.columns(4), stages, STAGE_CAPTIONS):
            with column:
                st.image(stage, caption=caption)

        with st.spinner("Encoding the animation..."):
            gif_stages = [downsample(stage, GIF_MAX_PX) for stage in stages]
            gif_frames = cross_fade_frames(gif_stages, FADE_FRAMES)
            gif_durations = frame_durations(len(gif_stages), FADE_FRAMES)
            gif_bytes = encode_gif(gif_frames, gif_durations)

        st.image(gif_bytes)

        timing_caption.caption(f"Processed in {time.time() - start:.2f}s")

        # Placed after the derived images rather than beside the original,
        # because the ShareAlike note refers to all of them.
        if st.session_state.use_sample:
            st.caption(SAMPLE_NOTICE)

        st.markdown("**Step-by-step guide**")
        st.caption(
            "Sends this photo, plus its measured value range and dominant "
            "temperature, to Claude for a written stage-by-stage guide."
        )

        if st.button("Generate step-by-step guide"):
            with st.spinner("Writing the guide..."):
                try:
                    writeup = generate_writeup(image_bytes, RUBRIC_VERSION, MODEL_NAME)
                # AuthenticationError stays first: it subclasses APIStatusError, so the
                # order of these handlers is what keeps a rejected key from being reported
                # as a generic status error.
                except anthropic.AuthenticationError:
                    show_saved_guide_or_error(
                        image_bytes,
                        "the API key was rejected",
                        "Invalid API key. Check the secret in Streamlit Cloud settings.",
                    )
                except anthropic.APIStatusError as e:
                    show_saved_guide_or_error(
                        image_bytes,
                        f"the API returned {e.status_code}",
                        f"API error: {e.message}",
                    )
                except anthropic.APIConnectionError:
                    show_saved_guide_or_error(
                        image_bytes,
                        "the API couldn't be reached",
                        "Couldn't reach the API. Check your internet connection.",
                    )
                else:
                    st.markdown(writeup)
