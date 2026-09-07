# Architecture

One page, on purpose. This covers what exists and the decisions that would be easy to get
wrong if you rebuilt this from scratch. Everything described here is built; if a claim in
this file ever outlives the code, that's a bug in the file.

## Components

- **`app.py`**: Streamlit UI and orchestration. Handles the upload and sample-photo
  inputs, calls into `imaging.py` for the actual math, and displays results. Streamlit
  reruns this entire file top to bottom on every user interaction (button click, file
  upload). That's the framework's execution model, not a bug, and it's why the Anthropic
  call is gated behind an explicit button instead of running automatically. It is also
  the only file in the project allowed to import Streamlit, which is what keeps every
  other module checkable from a plain Python shell, and the reason all four screens
  live here rather than in a module each.
- **`imaging.py`**: Pure image-math functions: numpy arrays in, numpy arrays out, zero
  Streamlit imports. Kept separate from `app.py` specifically so a function like
  `posterize` can be imported from a plain Python shell and run against a 4x4 array by
  hand, without booting a web server first. The line-drawing functions live here rather
  than in a module of their own even though they are a self-contained subsystem: they
  need `srgb_to_lab`, `posterize`, and the palette assignment, and `build_stages` needs
  them back, so a separate module would be a circular import for no gain.
- **`.streamlit/secrets.toml`** (local) / **Streamlit Cloud Secrets** (deployed): where
  the Anthropic API key lives. Never in code, never in git. `app.py` reads it once via
  `st.secrets["ANTHROPIC_API_KEY"]`.
- **`.streamlit/config.toml`**: the entire visual identity, as native Streamlit theme
  settings and no CSS. Injected CSS has to name Streamlit's generated class names, which
  are not a public API and change between releases, on a deployment whose upgrades are
  not ours to schedule; a theme file cannot break that way. Committed, unlike
  `secrets.toml` beside it, because it holds no secret and the deployed app needs it.
- **`gif.py`**: Encodes the four build stages into a looping GIF. Kept out of
  `imaging.py` because its output is GIF bytes through Pillow, not a numpy array back
  out; same standard otherwise, zero Streamlit imports.
- **`paints.py`**: The paint-tube reference table (measured CIE Lab masstones) and
  `nearest_paint`, matching a palette centroid to the closest tube by ΔE76. Same
  no-framework standard as `imaging.py`.
- **`rubric.py`**: The hand-written teaching rubric as a module-level string, plus
  `build_prompt`, which assembles the rubric, this photo's measured stats, and the
  output-shape instructions into the single prompt sent to the API. No Streamlit
  imports, so the exact prompt text can be printed and diffed from a terminal.
- **`demo_writeup.py`**: The saved step-by-step guide for the sample photo, served when
  the live API call fails, plus the rubric version, model, date, and photo hash it was
  generated under. Data only, no Streamlit imports; `app.py` owns the decision of when to
  show it and how to label it.
- **`guide.py`**: Cuts a written guide into one slice per stage, by finding the
  `STAGE_CAPTIONS` labels `rubric.py` asked the model for. Text in, text out, no
  Streamlit imports, so it is checkable against hand-typed strings rather than only
  through the browser. It returns `None` rather than a partly filled list when the labels
  are not all there in order, which is what lets `app.py` fall back to showing the guide
  whole instead of showing it with holes in it.
- **`assets/`**: The committed sample photo the "Try a sample photo" button loads, and
  the build-order GIF used in the README. Those two are the same Dead Vlei photograph,
  which is third-party CC BY-SA 4.0 material rather than project code; see the README for
  attribution. `assets/brand/` is the separate case: the logo files are this project's
  own work under the repository's own license, so nothing in there carries the photo's
  ShareAlike terms. `app.py` loads exactly one of them, the 260px web copy, which is both
  the header mark and the favicon.

## Data flow

A photo enters one of two ways, an upload or the sample-photo button, and both go through
`imaging.load_rgb`: Pillow opens the file, applies EXIF rotation, converts to RGB, and
resizes to a 1200px max dimension. That single decode path is deliberate, so the pixels
shown on screen and the pixels sent to the model are provably the same array.

From there the resized photo feeds four independent paths built from the *same* source
image, never chained off each other:
1. Grayscale → posterize → value study
2. Downsample → k-means in Lab space → sorted centroids → nearest-paint-tube match →
   palette
3. All four build stages generated in parallel from the source, then cross-faded → GIF.
   Stage 1 is also the image shown at the top of the page beside the original, computed
   once and reused rather than built twice
4. Photo + measured stats (value range, dominant temperature) + a hand-written teaching
   rubric → one Anthropic API call → written step-by-step

Path 3's stages depend on path 2's palette: the drawing outlines the shapes the palette
clusters define, and stages 2 and 3 paint with the palette's own six colors. That is the
one place the four paths touch, and it is deliberate. The filmstrip showing colors the
swatch panel does not is the version where the two outputs look unrelated.

Paths 1 through 3 run automatically on load, all inside one `prepare_photo` cached on
(image bytes, max dimension), so they run once per photo rather than once per rerun. Path
4 runs only on an explicit button click, and its result is cached on (image bytes, rubric
version, model). Path 4 is also the only one that can fail, since it is the only one that
leaves the machine, so it is the only one with a fallback: see demo mode below.

## Decisions that would be easy to get wrong

**Stages are computed, not generated, and this is the central bet of the whole app.** A
model asked to generate "stage 2 of this painting" produces *a picture of* a stage 2: it
invents detail, drifts off the original composition, and gives a different answer on
every run. A stage produced by posterizing and color-quantizing the *same* source photo
is provably still that photo, just with information removed, which is what an
underpainting actually is. It's also free, instant, and deterministic. Use a model where
the task needs judgment (the rubric write-up); use arithmetic where it doesn't.

**The stages open on a drawing, and the value studies are a checking tool rather than a
stage.** Until S12 the filmstrip ran values, color masses, soft focus, full detail, and
testing it on non-painters found they could not interpret the value study or say what to
do with it. Two things were wrong. The value study had no caption anywhere on the page
while every other output explained itself, so the test measured a presentation failure as
much as a content failure. And `rubric.py` had been saying since S9 to get the drawing and
the proportions down before any value work, which the images never showed: the text and
the pictures disagreed. So the drawing became stage 1, the progression stays in color
after it, and the value studies moved below the build order with a caption framing them as
the thing to hold your own block-in against.

The soft-focus stage was cut to make room rather than extending the filmstrip to five,
because the GIF's frame budget is frame count times frame size and that was already
measured. It is not a loss: a Gaussian blur was always standing in for "detail not yet
resolved," which is not an operation a painter performs on a canvas.

**A line is kept only where the two sides of it are genuinely different colors.** The
drawing unions two sources that fail in opposite directions. Region outlines, taken from a
label map of palette cluster crossed with value band, can only draw borders between large
areas, so anything thin or internal to one area is invisible to them. Canny edges on a
bilateral-filtered copy see exactly those and little else. Both then get the test neither
had alone: the two regions' own mean Lab values must differ by at least ΔE 15, the same
distance the paint matching uses. Without it the edge pass misses the sample photo's dune
ridge entirely, because a gradual sky-to-sand transition has no sharp gradient anywhere,
and the region pass invents wobbling lines across a smooth studio backdrop, because a
smooth backdrop still gets cut into clusters and every cut becomes a border.

**Posterize anchors its endpoints at true black and true white.** The step between levels
is `255 / (n_levels - 1)`, not `255 / n_levels`. The tempting version, integer division,
`(gray // 85) * 85` for three levels, produces bands that look fine but top out at 170,
not 255. A value study with no white in it is a muddy picture wearing a value study's
clothes.

**No tunable UI.** Level counts, cluster counts, and blur radii are hardcoded to curated
defaults rather than exposed as sliders. A slider quadruples the testing surface and lets
a visitor find the ugliest output the app can produce; a curated pair of studies is a
better demo than a configurable one.

**The model call is gated behind a button, never automatic.** Streamlit reruns the whole
script on every interaction, so an automatic call on upload would mean an unauthenticated
public page can trigger unbounded API spend. One click, one call.

That constraint is what shapes the step-at-a-time build order added in S14. The four
stages used to be four panels in a row; they are now one panel with Back and Next, which
turns a visit from near-zero clicks into four to seven, and every one of those clicks is a
full rerun of this file. Three things follow, all of them measured rather than assumed.
The call stays inside the button's own `if` body, so it fires on the click rerun and no
other: instrumented across one Generate click and twenty navigation and photo-switch
clicks, the API was called exactly once. The guide is held in session state rather than
recomputed, because a button reports True for a single rerun and the guide would otherwise
vanish on the visitor's next click. And the guide is stored beside a hash of the photo it
was written from, so switching photos drops it instead of describing a dead tree beside
somebody's kitchen: the same wrongness demo mode's `matches_sample` prevents, reached from
a different direction.

**Four screens, and the current one is a value in session state.** Streamlit has no
router. The two ways to get separate pages out of it are `st.navigation` with a `pages/`
directory, or a "which screen am I on" value that the whole script branches on once, at
the bottom. This app takes the second. `st.navigation` wants each page's code in a module
of its own, which would mean more than one file importing Streamlit and would break the
rule above that keeps every other module runnable from a plain shell. It also gives each
screen a URL, and the only thing a URL buys here is a way to land a stranger on the
tutorial screen for a photo that was never uploaded. With session state, the only path
onto a screen is a button on the screen before it, and the guard at the bottom of
`app.py` sends anything that needs a photo and has none back to the upload screen.

The cost is real and was accepted rather than talked around: `app.py` is now the router
as well as the UI, so it is one long file, and the screens cannot be linked, bookmarked,
or reached with the browser's own Back button.

The photo bytes move between screens in session state for the same reason. Streamlit
discards the state of any widget a rerun did not draw, and the uploader is only drawn on
one of the four screens, so the file it returned cannot be read from anywhere else. What
is stored is the raw bytes, which is also the cache key for `prepare_photo`, so a screen
change costs a cache lookup rather than a decode.

**Back and Next set state through `on_click` callbacks, not from inside an `if
st.button(...)` body.** A button only reports True during the rerun it was clicked in, and
by then the wizard has already read the step number and drawn a stage, so assigning in the
body moves the visitor one rerun late: Next appears dead until some unrelated click
happens to rerun the page. S12 shipped that bug twice in the supplies step, since removed,
where it initially read as working only because the checks around it were clicking other
things in between. The ends are disabled rather than hidden, so the controls keep the same
shape at every step and nothing shifts position under a thumb; measured at a 375px
viewport, the panel and both controls sit at the same y on all four steps.

The palette panel is drawn under the step's guide slice, inside the same function, so it
is present on all four stages rather than in a section of its own above them. That costs
nothing in page height because only one stage is drawn per rerun, and it is what makes the
tube names a reference a painter can keep looking at while mixing.

**The build-order GIF shares one color table across every frame instead of letting
Pillow quantize each frame separately.** Pillow's default is a per-frame palette, which
makes consecutive frames pick slightly different color tables and the animation flicker
on loop: invisible in a single screenshot, obvious in playback. `gif.py` samples rows from
every frame into one composite image, quantizes that once, and reuses the result as every
frame's palette at save time, so the file's GIF color tables actually agree frame to frame.

Those rows are spread evenly down each frame, not taken as a strip off the top, and that
detail is load-bearing. A top strip only represents a frame if the frame is uniform
vertically. S12's line drawing is not: the sample photo's drawing has nothing but flat
toned ground across its top quarter, so pure black never reached the quantizer, got no
palette entry, and every line in the animation's first frame rendered in the sky's dark
blue. The fix costs file size, since a palette that fits the whole frame gives
Floyd-Steinberg more near-neighbors to dither between and dithered flat areas compress
worse: the sample photo's GIF went from 86KB to 212KB.

That was checked against the thing S8 actually cared about, a phone on cell data, rather
than accepted or refused on the number alone. Re-measured in S19 on the tutorial screen,
which is the only screen the animation appears on: it is 207KiB of the 836KiB of images
that screen ships, so about a quarter of them, and the original photo alone is 505KiB.
That is the same file as the 212KB above, counted in KiB rather than decimal kB, which is
why one number is smaller than the other.

The share is larger than the 12% this file recorded through S18, and nothing about the GIF
changed to make it so. S14 did: replacing four stage panels in a row with one stage per
screen deleted most of the page the animation used to be a small fraction of. The
conclusion never rested on the percentage, so it survives intact. S8's concern was a 1.1MB
animation that was the dominant asset, and at 207KiB against a 505KiB photo the animation
is not the largest thing on its own screen. So dithering stays on, which is S8's decision,
not a default nobody rechecked. Turning it off was measured too and recovers about a
quarter of the size.

**An upload is refused on its header, before anything is decoded.** `load_rgb` reads
`Image.open(...).size`, which Pillow fills in from the header without allocating a pixel
buffer, and raises `UnusableImageSize` from there. Placing that check after the decode
would report the same error and prevent nothing, which is the entire point of it.

The failure it prevents is not a malformed file. It is a valid one: Pillow's own
decompression-bomb check raises only above *twice* its `MAX_IMAGE_PIXELS`, so a
13300x13300 PNG sits just under the threshold, decodes without complaint, and took peak
RSS to 932MiB when measured. That file is 0.16MB on disk. Streamlit Cloud's free tier has
roughly 1GB, so a small upload nobody would look at twice is an out-of-memory kill of a
public page. `MAX_SOURCE_PIXELS` is 50 million because the measured cost is about 3.5MiB
of peak RSS per megapixel and 50MP still clears every phone camera on sale.

There is a floor for a different reason. `extract_palette` cannot find `PALETTE_SIZE`
clusters in fewer than six pixels and raises a `ValueError` saying so, and until S19
nothing caught it: `prepare_photo` wraps only `load_rgb` in `UnreadableImage`, so a 2x2
PNG decoded fine and then put a Python traceback on the page. Six is where it broke;
`MIN_SOURCE_DIMENSION` is 64 because a 16x16 favicon runs the whole pipeline and produces
a palette of single pixels.

**Size failures and decode failures are two errors, not one.** `UnreadableImage` means
the bytes are not an image and its message says to try a JPEG, PNG, or HEIC. That advice
is actively wrong for someone whose JPEG is valid and merely enormous, so
`UnusableImageSize` carries its own text naming the actual size and the actual limit, and
`prepare_photo` re-raises it rather than flattening it into the decode error. The same
reasoning that kept `UnreadableImage` from being a bare `except` around the whole
pipeline applies one level down.

**Palette matching runs in Lab space, not RGB.** RGB numeric distance doesn't track how
different two colors *look*. Two pairs the same distance apart in RGB can be visually
nothing alike. Lab is built so equal distances look equally different, which is the
property a nearest-paint match actually needs. The reported ΔE numbers run large because
the reference table is masstones (paint straight from the tube) while most of a photo is
mid-values and tints; see the in-app "How the tube names are chosen" note for the same
caveat aimed at a user rather than a reviewer.

**One decode path, and the sample photo uses it too.** `imaging.load_rgb` is the only
place bytes become an array. The sample-photo button reads a committed file and hands
those bytes to the same function an upload goes through, so the sample exercises the real
pipeline rather than a shortcut around it, and a test script can drive the whole app by
clicking one button.

**Demo mode is scoped to one photo and says so on screen.** A public URL with no
authentication and a small prepaid balance will eventually run out of credits, so the
written guide is the one output that can turn into a red error box in front of a visitor.
When the API call raises, `app.py` serves the saved guide from `demo_writeup.py` instead,
with a note saying the live call is unavailable and that this text was generated earlier.
Two constraints shape the rest of it. It only covers the sample photo, decided by hashing
the bytes on the page against the photo the guide was written from, because there is
nothing pre-generated for a stranger's upload and a guide describing a dead tree next to
someone's portrait would be worse than an error; an upload therefore gets the error and a
pointer at the sample. And the saved guide records the rubric version it came from, so a
bumped `RUBRIC_VERSION` with a stale saved guide fails `test_demo_writeup.py` and changes
the on-screen note, rather than quietly teaching a rubric the app no longer sends.

**The written guide is split on the labels the prompt asked for, and shown whole when it
can't be.** `rubric.OUTPUT_INSTRUCTIONS` asks for four steps headed by the exact
`STAGE_CAPTIONS` strings, so the split is finding those four labels and cutting between
them. What the prompt asks for and what comes back are still different questions, so the
splitter was written against real replies rather than against the instruction: three were
checked, the saved sample guide plus live calls on two other photos, and all three used
the bare label on its own line. A model is free to wrap that same label in a header or
bold on any future call, so the match is made on a line stripped of markdown decoration.
The failure mode being designed against is silent text loss, since a splitter that drops a
paragraph when formatting shifts is worse than the single block it replaced. Hence nothing
is discarded (text above the first label joins the first slice) and failure is total
rather than partial (a missing or out-of-order label returns `None`, and `app.py` prints
the guide unsliced under a note saying so). The saved fallback guide goes down this same
path rather than around it.

**The supplies step was removed in S16, and the two state lessons from it outlived it.**
It asked what you were painting on and named the paint that suits it, sitting between the
photo and the painting steps. It is gone because the website-flow outline does not have a
place for it, not because it was broken. Both bugs found while building it are the same
shape as bugs this project has had elsewhere, so they are worth keeping written down.

Its skip and restore buttons assigned through `on_click` callbacks rather than inside an
`if st.button(...)` body. A button only reports `True` during the rerun it was clicked in,
by which point the function has already chosen which branch to draw, so assigning in the
body applies the change one rerun late. It read as working at first only because the checks
around it clicked other things in between, which is exactly how the S10 sample-button bug
hid, and it is why Back and Next are callbacks today.

Its chosen surface was kept in an app-owned session key and fed back as the radio's
starting index. Streamlit discards the state of any widget a rerun did not draw, so
collapsing the step and reopening it reset the answer to the first option: a visitor says
they are working on watercolor paper, skips, reopens, and is being told to buy oils. Any
future widget that can be drawn on some reruns and not others has the same problem.

**Testing the upload path itself.** The OS file-picker dialog cannot be driven by browser
automation, which is true and was for several sessions mistakenly treated as meaning the
upload path could not be tested at all. It can: assign a `DataTransfer`'s `files` to the
`input[type=file]` element and dispatch a `change` event, and Streamlit processes it as a
genuine upload. This matters beyond convenience. The difference between "the visitor just
uploaded this" and "that file has been sitting in the widget for several reruns" is
invisible to `if uploaded_file is not None`, because `st.file_uploader` returns the same
file every rerun until it is cleared, and that gap is exactly where a real bug lived. Any
change to the upload-versus-sample logic should be tested in both orderings, not just from
a clean page.
