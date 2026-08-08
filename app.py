import base64
import hashlib
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
from guide import split_guide
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

# On every screen, not just the front page. It is the only thing that says where
# you are once the upload control is behind you, and there is no browser URL doing
# that job: the screens are session state, not routes. Charlotte's own title and
# logo go here when she designs them.
st.title("Underpainting")

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
# "above" until the page became four screens, which made it false on three of them:
# the stage screens have no value study above them and the studies screen has no
# animation. Naming the derived work instead of pointing at it is true wherever the
# notice lands, and it has to be true on all of them, since the ShareAlike obligation
# follows every derived image and they are no longer all on one page.
SAMPLE_NOTICE = (
    f"Sample photo by Diego Delso, [delso.photo](https://delso.photo), "
    f"[source]({SAMPLE_SOURCE_URL}), licensed "
    f"[CC BY-SA 4.0]({SAMPLE_LICENSE_URL}). Resized from the original, and "
    "modified further by the value studies, build stages, and animation this app "
    "derives from it, which are themselves CC BY-SA 4.0."
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

# PLACEHOLDER TEXT, and Charlotte's to replace. The website-flow outline asks for a
# tutorial page and says the directions are hers to write later, so this constant is a
# scaffold for her words rather than a decision about what they say. These four are
# true of the app as it stands, because main stays deployable and a live page showing
# obvious filler is not deployable. This constant is the only place to change them.
TUTORIAL_STEPS = (
    "Work through the four stages in order. Each one is the same photo computed a "
    "different way, and each is a thing to finish before you move to the next.",
    "Squint at the stage on screen rather than at your photo. The stages throw away "
    "detail on purpose, and what is left is the part that has to be right.",
    "Keep the palette under each stage in view while you mix. It names the closest "
    "tube to each of the photo's six main colors, ordered by how much of the canvas "
    "each one covers.",
    "Generate the written guide below if you want words alongside the pictures. It is "
    "optional, and the stages work without it.",
)

# Bump this by hand whenever the prompt text changes. It's part of the cache
# key below specifically so an edited rubric produces a fresh writeup instead
# of silently returning a stale cached one for the same photo.
#
# "The prompt text" is rubric.RUBRIC and also imaging.STAGE_CAPTIONS, which
# rubric.py interpolates into the step labels it asks the model for. v3 is
# the S12 stage redesign: both changed at once, since renaming the stages is
# what forced the rubric rewrite.
RUBRIC_VERSION = "v3"


class UnreadableImage(Exception):
    """The uploaded bytes could not be decoded as an image.

    Named rather than caught as a bare Exception around the whole pipeline, so the
    on-screen "try a JPEG, PNG, or HEIC" message stays attached to the one failure it
    actually describes. Everything else in prepare_photo runs on an array that already
    decoded, and telling a visitor their file is unreadable when the palette clustering
    is what broke would send them off fixing the wrong thing.
    """


@st.cache_data(show_spinner=False)
def prepare_photo(image_bytes, max_dimension):
    """Every image computed from the photo, done once per photo instead of once per rerun.

    Streamlit reruns the whole script on every interaction, and as of S14 the build order
    is a wizard, so a visit is 4 to 7 clicks rather than the near-zero it used to be.
    Measured before this was added: 0.49 to 0.55s per rerun across the three test photos
    (0.44 to 0.50s for the studies, palette, and stages; 0.05 to 0.06s for the GIF). That
    is a few seconds of recomputing images that cannot have changed, since nothing a Next
    click touches is an input to any of them.

    Keyed on (image_bytes, max_dimension) rather than on the decoded array, for the same
    reason generate_writeup is keyed on bytes: they are two simple hashable values, and
    the rgb array is a pure function of them, so hashing the array would be the same
    cache key arrived at slower. Everything downstream is likewise deterministic given
    that array, which is what makes one entry per photo correct rather than merely cheap.

    Args:
        image_bytes: raw bytes of the uploaded file, unresized.
        max_dimension: longest edge to resize to, see MAX_DIMENSION.

    Returns:
        Dict of the rgb array, the two value studies, the palette, the four stages, the
        encoded GIF, and "seconds", the wall time this took. The timing is measured in
        here rather than around the call on purpose: it is the cost of processing the
        photo, which is what the caption on screen claims it is. Timed at the call site
        it would read as 0.01s on every rerun after the first, which is a true statement
        about a cache hit and a false one about the pipeline.
    """
    started = time.time()

    try:
        rgb = load_rgb(image_bytes, max_dimension)
    except Exception as e:
        raise UnreadableImage from e

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    palette = extract_palette(rgb)

    # Built here rather than beside the filmstrip, because stage 1 is also the
    # image at the top of the page and computing the drawing twice would double
    # the one expensive step on the page.
    stages = build_stages(rgb, palette)

    gif_stages = [downsample(stage, GIF_MAX_PX) for stage in stages]
    gif_frames = cross_fade_frames(gif_stages, FADE_FRAMES)
    gif_durations = frame_durations(len(gif_stages), FADE_FRAMES)

    return {
        "rgb": rgb,
        "coarse_study": posterize(gray, COARSE_LEVELS),
        "fine_study": posterize(gray, FINE_LEVELS),
        "palette": palette,
        "stages": stages,
        "gif_bytes": encode_gif(gif_frames, gif_durations),
        "seconds": time.time() - started,
    }


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


def saved_guide_or_error(image_bytes, short_reason, error_message):
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

    Returns rather than renders, as of S14. The saved guide was written as one block and
    used to be printed as one, but it is a real reply in the same four-label shape as a
    live one, so it now goes down the identical path: stored, split, and shown a stage at
    a time beside the matching image. A fallback that rendered itself would have been the
    one guide on the page that ignored the wizard.

    Args:
        image_bytes: the bytes currently being displayed, sample or upload.
        short_reason: visitor-facing phrase for why the live call didn't happen.
        error_message: the operator-facing message, kept from S4, for the no-fallback case.

    Returns:
        A guide dict in the shape described at store_guide.
    """
    if not demo_writeup.matches_sample(image_bytes):
        return {
            "text": None,
            "notice": error_message,
            "notice_kind": "error",
            "caption": (
                "There's nothing saved for a photo the app hasn't seen before. The sample "
                "photo above has a saved guide that works without the API, if you want to "
                "see what this output looks like."
            ),
        }

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

    return {
        "text": demo_writeup.WRITEUP,
        "notice": (
            f"The live model call is unavailable right now ({short_reason}), so this is a "
            "saved guide for the sample photo rather than one written just now. Everything "
            "else on this page was still computed live from the photo."
        ),
        "notice_kind": "info",
        "caption": provenance,
    }


def store_guide(guide):
    """Hold a generated guide across the reruns Next and Back cause.

    The button that produces a guide reports True for exactly one rerun, so without
    this the guide would appear and then vanish on the visitor's very next click. It
    is kept beside the hash of the photo it was written from, because the failure that
    matters here is not losing it but showing it beside the wrong picture: the guide
    names what is actually in the image, so serving one photo's guide next to another
    is the same error demo_writeup.matches_sample exists to prevent, arrived at from a
    different direction.

    A guide dict is: "text", the writeup itself or None when the call failed with
    nothing saved to fall back on; "notice" and "notice_kind", an on-screen info or
    error line or None; and "caption", the provenance line under it or None.
    """
    st.session_state.guide = guide
    st.session_state.guide_photo = st.session_state.photo_key


def show_guide_notice(guide):
    """The info or error line for a guide that did not come back live, if there is one."""
    if guide["notice"]:
        (st.error if guide["notice_kind"] == "error" else st.info)(guide["notice"])
    if guide["caption"]:
        st.caption(guide["caption"])


def _go_to(screen):
    """Move to another screen. Every navigation control on the app goes through here."""
    st.session_state.screen = screen


def _step_back():
    """One stage back, or off the front of the wizard and onto the tutorial."""
    if st.session_state.stage_step == 0:
        st.session_state.screen = "tutorial"
        return
    st.session_state.stage_step -= 1


def _step_forward():
    """One stage on, or off the end of the wizard and onto the value studies."""
    if st.session_state.stage_step == len(STAGE_CAPTIONS) - 1:
        st.session_state.screen = "studies"
        return
    st.session_state.stage_step += 1


def _back_to_last_stage():
    """From the value studies back into the wizard, at the stage it was left on.

    The step is set rather than assumed. Reaching the studies means stepping off the
    end of the wizard, so the step is already the last one every ordinary way in, but
    "already true" and "made true" differ the first time somebody adds another way to
    get here.
    """
    st.session_state.stage_step = len(STAGE_CAPTIONS) - 1
    st.session_state.screen = "stages"


def show_palette_reference(palette):
    """The palette and its tube names, small, under whichever stage is on screen.

    This used to be a full section of its own above the build order. It is a reference
    a painter checks while mixing, not something read once and scrolled past, so it now
    sits under every stage instead, which is what the website-flow outline asks for.

    Repeating it on all four stages costs nothing in page height, because the wizard
    draws exactly one stage per rerun. What it does cost is space under each stage, so
    the panel is smaller than the section it replaces, and the compaction is where the
    measuring went rather than the move itself. Streamlit stacks st.columns below its
    mobile breakpoint, so on a phone this is six cells one under another and the cost
    per cell is what matters: the old section spent three elements on each, a 90px
    swatch, a bold name, and a two-line caption, and Streamlit's block gap between
    them is roughly as tall as the text. Folding the name and numbers into the image's
    own caption makes each cell one element instead of three. Measured at a 375px
    viewport on the sample photo: 1,550px as a section, 889px moved but otherwise
    unchanged, 690px like this, which is inside one 812px phone screen. The paragraph
    explaining the Lab clustering moved into the expander for the same reason.

    Nothing was dropped to get there. Every number the section showed is still on
    screen, which is deliberate: the ΔE figures are the honest part of this panel, and
    a compaction that quietly deleted them would be selling the match as better than
    it is.

    Args:
        palette: the swatch dicts from imaging.extract_palette, ordered by coverage.
    """
    st.markdown("**Palette and closest tube**")

    for column, swatch in zip(st.columns(len(palette)), palette):
        # The swatch is the photo's own centroid, never a render of the
        # paint measurement. Showing the paint's color would quietly claim
        # the match is exact.
        paint_name, delta_e = nearest_paint(swatch["lab"])
        with column:
            # Two trailing spaces are a markdown hard line break, which st.image
            # captions honor. The tube name gets its own line because it is the
            # thing being looked up; the measurements sit under it.
            st.image(
                np.full((28, 140, 3), swatch["rgb"], dtype=np.uint8),
                caption=(
                    f"**{paint_name}**  \n"
                    f"{swatch['share'] * 100:.1f}% · {swatch['hex']} · "
                    f"ΔE {delta_e:.0f}"
                ),
            )

    with st.expander("How the tube names are chosen"):
        st.markdown(
            "Clustered in Lab so the groupings match colors a painter would mix "
            "as one, ordered by how much of the canvas each covers. Each swatch "
            "shows the photo's own color, with the tube you would reach for to "
            "mix it.\n\n"
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


def show_stage_wizard(stages, palette, slices, hint):
    """The build order one stage at a time, instead of four panels in a row.

    Both controls set state through on_click callbacks rather than by assigning inside
    an `if st.button(...)` body, the same as every other navigation control in this
    file and for the same measured reason: a button only reports True during the rerun
    it was clicked in, and by then this function has already read stage_step and
    decided which stage to draw.
    Assigning in the body applies the move one rerun late, so Next appears to do nothing
    until some unrelated click happens to rerun the page. S12 shipped that bug twice in
    the supplies step, since removed; this wizard repeats the pattern four steps wide,
    which is the reason the S14 brief called state ordering the real risk here rather
    than the widget code.

    The ends no longer dead-end. Until this session Back on the drawing and Next on
    full detail were disabled, because there was nothing on either side of them; now
    the wizard is one screen of four, so Back on stage 1 returns to the tutorial and
    Next on stage 4 goes on to the value studies. Both controls are therefore always
    live, which keeps the promise the disabled version was making in a different way:
    the row is the same two same-sized buttons in the same place on every stage, and
    nothing moves under a thumb.

    Args:
        stages: the four stage arrays from imaging.build_stages.
        palette: the swatch dicts, passed straight through to show_palette_reference.
        slices: one guide slice per stage, or None when there is no guide to show or it
            could not be split. Never a partially filled list; see guide.split_guide.
        hint: a short line to show under the panel when there is no slice, or None.
    """
    step = st.session_state.stage_step

    st.caption(f"Step {step + 1} of {len(STAGE_CAPTIONS)}")
    st.image(stages[step], caption=STAGE_CAPTIONS[step])

    # Back and Next sit directly under the image rather than under the text, because
    # Streamlit stacks st.columns vertically below its mobile breakpoint: with the
    # guide slice in between, a phone visitor would scroll past a full step of writing
    # to find the control that advances it. Under the image they are within a thumb's
    # reach of the thing they move in both layouts.
    # width="stretch" rather than use_container_width=True: the latter is
    # deprecated as of the pinned Streamlit 1.60 and warns on every rerun.
    # Stretched so the two controls are the same size and a thumb cannot land
    # between them on a phone.
    back_column, next_column = st.columns(2)
    with back_column:
        st.button("Back", on_click=_step_back, width="stretch")
    with next_column:
        st.button("Next", on_click=_step_forward, width="stretch")

    if slices is not None:
        st.markdown(slices[step])
    elif hint:
        st.caption(hint)

    # Last, so it sits under the whole step rather than between the picture and the
    # writing about it. The outline's wording is "right below the painting steps on
    # each stage", and the writing is part of the step.
    show_palette_reference(palette)




def show_upload_screen():
    """Step 0: the front page. One thing to do, and a way to do it without a photo.

    The outline asks for a big Upload button, a line of copy directly above it, and a
    tip that high contrast and simple shapes work best. The sample button is not in the
    outline and is here anyway, because a front page whose only action needs a photo
    file is a dead end for the visitor this app is most often shown to: someone opening
    the link on a laptop with no reference photo to hand. Two clicks to real output
    instead of none.
    """
    st.write("Get customized painting instructions from start to finish")

    # Keyed, because the callback has to read the widget's value during the callback
    # rather than from the return value further down: by the time the script reaches
    # the return value this screen may not be the screen being drawn any more.
    st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "heic", "heif"],
        key="uploader",
        on_change=_use_uploaded_photo,
    )
    st.caption("Tip: high contrast and simple shapes are best.")

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    st.caption("No photo handy?")
    st.button("Try a sample photo", on_click=_use_sample_photo)


def show_tutorial_screen(photo):
    """Step 2: one scrolling page between the upload and the painting steps.

    The outline's shape, in its order: the directions, then a GIF of the visitor's own
    image, then a "Let's start!" button at the bottom. The photo and its line drawing
    sit above the directions rather than being dropped, because this is now the only
    screen that shows the visitor what they uploaded, and the drawing caption is the
    S12 answer to non-painters not knowing what the first output was for.

    The Generate button lives here for a mechanical reason, not a design one. A click
    is reported for exactly one rerun, so a button drawn after the panel that reads its
    result delivers the guide a rerun late. Putting it on the screen before the stages
    makes that ordering structural instead of something to remember.
    """
    original_column, drawing_column = st.columns(2)
    with original_column:
        st.image(photo["rgb"], caption=st.session_state.source_caption)
    with drawing_column:
        st.image(photo["stages"][0], caption="Line drawing")

    # The grayscale copy that sat here until S12 was the one output on
    # the page with nothing to say about itself, and testing on
    # non-painters found they could not say what to do with it. The
    # drawing is the step they were actually missing, and it is the one
    # the rubric has always said comes first.
    st.caption(
        "The biggest shapes and the real edges between them. This is the first "
        "thing to get down on the canvas, before any paint: an edge only counts "
        "here if the two sides of it are genuinely different colors, so what "
        "survives is structure rather than texture."
    )

    # Pixel dimensions are noise to a visitor; processing time is a real
    # signal to a technical one ("this ran in a fraction of a second"),
    # so it stays and the dimensions go.
    #
    # The number comes out of prepare_photo rather than from a timer around
    # it. Two ways this has been wrong before, and the honest number avoids
    # both: timing only as far as the palette reported 0.12s of a 0.25s
    # pipeline, and timing at the call site would now report a cache hit on
    # every click after the first, which is a true number about the wrong thing.
    st.caption(f"Processed in {photo['seconds']:.2f}s")

    st.markdown("**How this works**")
    for number, step in enumerate(TUTORIAL_STEPS, start=1):
        st.markdown(f"{number}. {step}")

    st.image(
        photo["gif_bytes"],
        caption="All four stages in one loop, before stepping through them.",
    )

    st.caption(
        "Optional. Sends this photo, plus its measured value range and dominant "
        "temperature, to Claude for a written guide, one part beside each stage. "
        "This is the only thing in the app that calls a model."
    )
    if st.button("Generate step-by-step guide"):
        with st.spinner("Writing the guide..."):
            _generate_and_store_guide(st.session_state.image_bytes)

    guide = current_guide()
    if guide is not None:
        show_guide_notice(guide)
        if guide["text"]:
            st.caption(
                "Written. It is split across the four stages, so it appears as you "
                "step through them."
            )

    back_column, start_column = st.columns(2)
    with back_column:
        st.button(
            "Back", on_click=_go_to, args=("upload",), width="stretch"
        )
    with start_column:
        st.button(
            "Let's start!",
            on_click=_go_to,
            args=("stages",),
            type="primary",
            width="stretch",
        )

    if st.session_state.use_sample:
        st.caption(SAMPLE_NOTICE)


def show_stages_screen(photo):
    """Step 3: one painting stage per screen, with its guide slice and the palette."""
    guide = current_guide()

    # The notice travels with the text it is about, rather than being left behind on
    # the screen where the button was clicked. A saved guide labelled as saved on one
    # screen and shown unlabelled on the next is the same guide passed off two
    # different ways, and the label is the honest half. It renders once per screen for
    # the same reason the palette does: only one stage is ever drawn.
    if guide is not None:
        show_guide_notice(guide)

    slices = None
    unsplit = None
    if guide is not None and guide["text"]:
        slices = split_guide(guide["text"], STAGE_CAPTIONS)
        # A guide the split could not account for is still shown, whole and
        # in one piece, rather than dropped or shown with holes in it. The
        # wizard keeps working on the images either way, so the worst case
        # here is the page the visitor would have had before this session.
        if slices is None:
            unsplit = guide["text"]

    show_stage_wizard(
        photo["stages"],
        photo["palette"],
        slices,
        hint=None if guide is not None else (
            "Stepping through the stages does not need the guide. Generate it on "
            "the previous screen if you want the writing beside each one."
        ),
    )

    if unsplit is not None:
        st.caption(
            "The guide did not come back in the four labelled steps this page "
            "splits on, so here it is whole."
        )
        st.markdown(unsplit)

    if st.session_state.use_sample:
        st.caption(SAMPLE_NOTICE)


def show_studies_screen(photo):
    """The last screen: what to hold your own block-in against.

    After the stages rather than before them, which is the S12 decision kept intact
    through the page split. Non-painters shown the old page could not say what to do
    with a value study sitting second from the top with no caption; it is a thing you
    compare a block-in against, which only means anything once there is a block-in.
    """
    st.markdown("**Value studies**")
    st.caption(
        "Not a stage to copy. This is what to check your own block-in against: "
        "squint at your canvas until the detail drops away, and compare the "
        "masses that are left against these. Aim for roughly this many, 3 on "
        "the roughest pass and 5 as it refines, rather than an arbitrary number."
    )

    coarse_column, fine_column = st.columns(2)
    with coarse_column:
        st.image(photo["coarse_study"], caption=f"{COARSE_LEVELS}-value study")
    with fine_column:
        st.image(photo["fine_study"], caption=f"{FINE_LEVELS}-value study")

    back_column, restart_column = st.columns(2)
    with back_column:
        st.button(
            "Back", on_click=_back_to_last_stage, width="stretch"
        )
    with restart_column:
        st.button(
            "Start over", on_click=_go_to, args=("upload",), width="stretch"
        )

    # Placed after the derived images rather than beside the original,
    # because the ShareAlike note refers to all of them.
    if st.session_state.use_sample:
        st.caption(SAMPLE_NOTICE)


def current_guide():
    """The stored guide, but only if it was written from the photo currently loaded.

    A guide left over from a previous photo describes things that are not on screen,
    which is the same wrongness demo mode's matches_sample guards against, reached from
    a different direction. Checked here rather than at the point the guide is stored,
    because the two screens that show it are two callers and this is one answer.
    """
    guide = st.session_state.guide
    if guide is None or st.session_state.guide_photo != st.session_state.photo_key:
        return None
    return guide


def _generate_and_store_guide(image_bytes):
    """The one model call, and every way it is allowed to fail."""
    try:
        writeup = generate_writeup(image_bytes, RUBRIC_VERSION, MODEL_NAME)
    # AuthenticationError stays first: it subclasses APIStatusError, so the
    # order of these handlers is what keeps a rejected key from being reported
    # as a generic status error.
    except anthropic.AuthenticationError:
        store_guide(saved_guide_or_error(
            image_bytes,
            "the API key was rejected",
            "Invalid API key. Check the secret in Streamlit Cloud settings.",
        ))
    except anthropic.APIStatusError as e:
        store_guide(saved_guide_or_error(
            image_bytes,
            f"the API returned {e.status_code}",
            f"API error: {e.message}",
        ))
    except anthropic.APIConnectionError:
        store_guide(saved_guide_or_error(
            image_bytes,
            "the API couldn't be reached",
            "Couldn't reach the API. Check your internet connection.",
        ))
    else:
        store_guide({
            "text": writeup,
            "notice": None,
            "notice_kind": None,
            "caption": None,
        })


def _adopt_photo(image_bytes, source_caption, from_sample):
    """Take on a photo and move to the tutorial, or stay put and say why not.

    The decode is attempted here, inside the callback, rather than being left to the
    screen that renders next. A file Pillow cannot open should leave the visitor on the
    screen with the upload control on it, next to an error about the file they just
    picked; bouncing them to a tutorial page that renders an error instead of a photo
    is a worse version of the same information. prepare_photo is cached, so paying for
    the decode here means the tutorial does not pay for it again.

    A different photo invalidates the step position. The comparison is on the bytes
    rather than on which control was used, so it is also right for the case neither
    control can see: a visitor uploading a second copy of the file already loaded.
    """
    st.session_state.upload_error = None
    try:
        prepare_photo(image_bytes, MAX_DIMENSION)
    except UnreadableImage:
        st.session_state.upload_error = (
            "Couldn't read that file as an image. Try a JPEG, PNG, or HEIC photo."
        )
        return

    photo_key = hashlib.sha256(image_bytes).hexdigest()
    if photo_key != st.session_state.photo_key:
        st.session_state.photo_key = photo_key
        st.session_state.stage_step = 0

    st.session_state.image_bytes = image_bytes
    st.session_state.source_caption = source_caption
    st.session_state.use_sample = from_sample
    st.session_state.screen = "tutorial"


def _use_uploaded_photo():
    """The uploader changed: a new file, or the visitor clearing the one that was there.

    This has to be a callback rather than a plain `if uploaded_file is not None` check
    after the widget. st.file_uploader keeps returning the same file on every rerun
    until it is cleared, so a plain check cannot tell "the visitor just uploaded this"
    from "that file has been sitting there since three clicks ago". Reading it as the
    former made the sample button dead for anyone who had already uploaded something.
    Measured, not assumed, in S10.

    Now that the upload is a screen rather than a section, the callback carries the
    navigation too: the outline says an upload sends you straight to the tutorial.
    """
    uploaded = st.session_state.uploader
    if uploaded is None:
        return
    _adopt_photo(uploaded.getvalue(), "Your photo", from_sample=False)


def _use_sample_photo():
    _adopt_photo(
        SAMPLE_IMAGE_PATH.read_bytes(),
        f"Sample photo · {SAMPLE_CREDIT}",
        from_sample=True,
    )


# Streamlit has no router, so which screen you are on is a value in session state and
# the script branches on it once, at the bottom. The alternative, st.navigation and a
# pages/ directory, was weighed and turned down: it wants page code in modules of its
# own, which is a change to the one-UI-file rule in CLAUDE.md, and it gives each screen
# a URL, which here mostly means a way to deep-link a stranger into a tutorial page for
# a photo that was never uploaded. This way the only path onto a screen is a button
# that had to pass through the screen before it.
if "screen" not in st.session_state:
    st.session_state.screen = "upload"

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "source_caption" not in st.session_state:
    st.session_state.source_caption = None

if "upload_error" not in st.session_state:
    st.session_state.upload_error = None

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

if "stage_step" not in st.session_state:
    st.session_state.stage_step = 0

if "guide" not in st.session_state:
    st.session_state.guide = None

if "photo_key" not in st.session_state:
    st.session_state.photo_key = None

if "guide_photo" not in st.session_state:
    st.session_state.guide_photo = None

# Three of the four screens are meaningless without a photo, and session state outlives
# a lot: a Streamlit Cloud container recycling, a tab left open overnight. Rather than
# have each screen defend itself, anything that needs a photo and has none falls back
# to the one screen that can produce one.
if st.session_state.screen != "upload" and st.session_state.image_bytes is None:
    st.session_state.screen = "upload"

if st.session_state.screen == "upload":
    show_upload_screen()
else:
    # Cached, and already warmed by _adopt_photo, so the spinner is for the case this
    # is a cold container rather than for the ordinary click.
    with st.spinner("Reading your photo and building the stages..."):
        photo = prepare_photo(st.session_state.image_bytes, MAX_DIMENSION)

    if st.session_state.screen == "tutorial":
        show_tutorial_screen(photo)
    elif st.session_state.screen == "stages":
        show_stages_screen(photo)
    else:
        show_studies_screen(photo)
