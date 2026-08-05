# Architecture

One page, on purpose. This covers what exists and the decisions that would be easy to get
wrong if you rebuilt this from scratch. Everything described here is built; if a claim in
this file ever outlives the code, that's a bug in the file.

## Components

- **`app.py`**: Streamlit UI and orchestration. Handles the upload and sample-photo
  inputs, calls into `imaging.py` for the actual math, and displays results. Streamlit
  reruns this entire file top to bottom on every user interaction (button click, file
  upload). That's the framework's execution model, not a bug, and it's why the Anthropic
  call is gated behind an explicit button instead of running automatically.
- **`imaging.py`**: Pure image-math functions: numpy arrays in, numpy arrays out, zero
  Streamlit imports. Kept separate from `app.py` specifically so a function like
  `posterize` can be imported from a plain Python shell and run against a 4x4 array by
  hand, without booting a web server first.
- **`.streamlit/secrets.toml`** (local) / **Streamlit Cloud Secrets** (deployed): where
  the Anthropic API key lives. Never in code, never in git. `app.py` reads it once via
  `st.secrets["ANTHROPIC_API_KEY"]`.
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
- **`assets/`**: The committed sample photo the "Try a sample photo" button loads, and
  the build-order GIF used in the README. Both are the same Dead Vlei photograph, which
  is third-party CC BY-SA 4.0 material rather than project code; see the README for
  attribution.

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
3. All four build stages generated in parallel from the source, then cross-faded → GIF
4. Photo + measured stats (value range, dominant temperature) + a hand-written teaching
   rubric → one Anthropic API call → written step-by-step

Paths 1 through 3 run automatically on load. Path 4 runs only on an explicit button
click, and its result is cached on (image bytes, rubric version, model).

## Decisions that would be easy to get wrong

**Stages are computed, not generated, and this is the central bet of the whole app.** A
model asked to generate "stage 2 of this painting" produces *a picture of* a stage 2: it
invents detail, drifts off the original composition, and gives a different answer on
every run. A stage produced by posterizing and color-quantizing the *same* source photo
is provably still that photo, just with information removed, which is what an
underpainting actually is. It's also free, instant, and deterministic. Use a model where
the task needs judgment (the rubric write-up); use arithmetic where it doesn't.

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

**The build-order GIF shares one color table across every frame instead of letting
Pillow quantize each frame separately.** Pillow's default is a per-frame palette, which
makes consecutive frames pick slightly different color tables and the animation flicker
on loop: invisible in a single screenshot, obvious in playback. `gif.py` samples pixel
strips from every frame into one composite image, quantizes that once, and reuses the
result as every frame's palette at save time, so the file's GIF color tables actually
agree frame to frame.

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
pipeline rather than a shortcut around it. That is also what makes the sample button
usable as an end-to-end test: browser automation cannot open a native file picker, so
before it existed there was no way to drive the full app from a script.
