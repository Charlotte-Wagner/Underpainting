# Architecture

One page, on purpose. This covers what exists, what's planned, and the decisions that
would be easy to get wrong if you rebuilt this from scratch.

## Components

- **`app.py`** — Streamlit UI and orchestration. Handles upload, calls into `imaging.py`
  for the actual math, and displays results. Streamlit reruns this entire file top to
  bottom on every user interaction (button click, file upload) — that's the framework's
  execution model, not a bug, and it's why the Anthropic call is gated behind an explicit
  button instead of running automatically.
- **`imaging.py`** — Pure image-math functions: numpy arrays in, numpy arrays out, zero
  Streamlit imports. Kept separate from `app.py` specifically so a function like
  `posterize` can be imported from a plain Python shell and run against a 4x4 array by
  hand, without booting a web server first. Also holds palette extraction: downsample →
  k-means in Lab space → centroids sorted by canvas coverage.
- **`paints.py`** — The paint-tube reference table (measured Lab masstones) and
  `nearest_paint`, a nearest-neighbor match by Lab distance (ΔE) from a palette centroid
  to the closest tube. Same numpy-in/numpy-out standard as `imaging.py`.
- **`.streamlit/secrets.toml`** (local) / **Streamlit Cloud Secrets** (deployed) — where
  the Anthropic API key lives. Never in code, never in git. `app.py` reads it once via
  `st.secrets["ANTHROPIC_API_KEY"]`.
- **Planned, not yet built:** a stage-generator module (the four-image build-order
  sequence + GIF export), and the rubric prompt sent alongside the photo to the Anthropic
  API.

## Data flow

**Today:** upload → Pillow opens the file, applies EXIF rotation, converts to RGB, resizes
to a 1200px max dimension → converted to a numpy array → OpenCV converts to grayscale →
`imaging.posterize()` runs twice (3 levels, 5 levels) → Streamlit displays original,
grayscale, and both studies side by side.

**Planned, full pipeline:** the resized photo feeds four independent paths built from the
*same* source image, not chained off each other:
1. Grayscale → posterize → value study (built)
2. Downsample → k-means in Lab space → sorted centroids → nearest-paint-tube match →
   palette (built)
3. All four build stages generated in parallel from the source, then cross-faded → GIF
   (not built)
4. Photo + measured stats (value range, dominant palette, temperature) + a hand-written
   teaching rubric → one Anthropic API call → written step-by-step (not built)

## Decisions that would be easy to get wrong

**Stages are computed, not generated, and this is the central bet of the whole app.** A
model asked to generate "stage 2 of this painting" produces *a picture of* a stage 2 — it
invents detail, drifts off the original composition, and gives a different answer on
every run. A stage produced by posterizing and color-quantizing the *same* source photo
is provably still that photo, just with information removed — which is what an
underpainting actually is. It's also free, instant, and deterministic. Use a model where
the task needs judgment (the rubric write-up); use arithmetic where it doesn't.

**Posterize anchors its endpoints at true black and true white.** The step between levels
is `255 / (n_levels - 1)`, not `255 / n_levels`. The tempting version — integer division,
`(gray // 85) * 85` for three levels — produces bands that look fine but top out at 170,
not 255. A value study with no white in it is a muddy picture wearing a value study's
clothes.

**No tunable UI.** Level counts, cluster counts, and blur radii are hardcoded to curated
defaults rather than exposed as sliders. A slider quadruples the testing surface and lets
a visitor find the ugliest output the app can produce; a curated pair of studies is a
better demo than a configurable one.

**The model call is gated behind a button, never automatic.** Streamlit reruns the whole
script on every interaction, so an automatic call on upload would mean an unauthenticated
public page can trigger unbounded API spend. One click, one call.

**Palette matching runs in Lab space, not RGB.** RGB numeric distance doesn't track how
different two colors *look* — two pairs the same distance apart in RGB can be visually
nothing alike. Lab is built so equal distances look equally different, which is the
property a nearest-paint match actually needs. The reported ΔE numbers run large because
the reference table is masstones (paint straight from the tube) while most of a photo is
mid-values and tints — see the in-app "How the tube names are chosen" note for the same
caveat aimed at a user instead of a reviewer.
