# Underpainting

Underpainting turns a reference photo into a set of study aids for representational
painting: a simplified value study, a limited paint palette with real tube names, a
staged build-order preview, and a written step-by-step. It's for painting students
staring at a photo who don't know what order to build it in.

This is a work in progress, built in public across three weekends. **See
[Status](#status) below for exactly what's real today** before assuming a feature works.

**Live app:** _add the Streamlit Cloud URL here once deployed_

---

## Tech stack

| Piece | Why |
|---|---|
| **Streamlit** | Ships a working UI without writing frontend code, and deploys free to a public URL — the two things a 3-weekend solo portfolio project needs most. |
| **Pillow + pillow-heif** | Image I/O, EXIF rotation, and HEIC support — real photos come straight off a phone, sideways, sometimes in Apple's format. |
| **OpenCV (headless)** | Color-space conversion (grayscale now; Lab for perceptual color matching later). Headless build specifically — the normal `opencv-python` package expects desktop graphics libraries that don't exist on a cloud container. |
| **numpy** | The actual pixel math. Every image is an array; every transformation is arithmetic on that array. |
| **Anthropic API** | The one deliberately non-deterministic piece: turning a photo plus a hand-written painting rubric into a written, specific step-by-step. Everything else in the app is plain image processing on purpose — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why. |

## Setup

Starting from a fresh clone, on macOS/Linux (Windows notes inline):

```bash
# 1. Clone and enter the project
git clone https://github.com/Charlotte-Wagner/Underpainting.git
cd Underpainting

# 2. Create and activate a virtual environment
#    (Python 3.12 recommended, matching the deployed environment; 3.10+ works)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add an API key — only needed for the "Ask Claude something" button.
#    Everything else runs with no key at all. See .env.example for the
#    variable name; get a real key at console.anthropic.com.
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# 5. Run it
streamlit run app.py
```

Opens at `http://localhost:8501`. Every command above was run from a clean clone before
this README shipped — none of it is guessed.

## Running the checks

There's no automated test suite yet (see [Status](#status)). The image-math functions in
`imaging.py` are deliberately plain numpy in, numpy out, with zero Streamlit calls, so
they can be checked by hand against an array small enough to read by eye:

```bash
python -c "
from imaging import posterize, posterize_output_values
import numpy as np
gray = np.array([[0, 50, 128, 200, 255]], dtype=np.uint8)
print('posterize(gray, 3):', posterize(gray, 3).tolist())
print('possible 3-level values:', posterize_output_values(3).tolist())
"
```

Expected output: `posterize(gray, 3)` maps every pixel onto `[0, 128, 255]` — the point
being that the lightest value lands on true white, not short of it (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why that's the whole design decision).

## Status

**Working today:**
- Photo upload (JPEG, PNG, HEIC) with EXIF-safe rotation and resize to a 1200px max
  dimension
- Grayscale conversion
- Value study: posterize to 3 and 5 tonal levels, endpoints anchored at true black and
  true white
- The four-stage build-order filmstrip, computed independently per stage from the same
  source photo, and its cross-faded animated GIF
- Anthropic API connectivity, gated behind an explicit button (never fires on upload)
- Dev Container config for GitHub Codespaces

**In progress / not yet built:**
- Palette extraction (k-means clustering in Lab color space) and matching each color to a
  real paint tube name from a hand-built lookup table
- The rubric-driven written step-by-step — the one output that calls a model
- Automated tests
- Deployed public link (see the top of this README once it's live)

**Intentionally out of scope for v1:**
- User accounts, saved history, or any database
- Mobile app or native client
- Payments
- Live camera capture (upload only)

Cutting these isn't a gap — v1 is upload-a-photo, get-a-result, nothing more, on
purpose. Anything else goes in a running v2 list instead of into this branch.

## Screenshots

_Coming once the four core outputs (value study, palette, stage progression, write-up)
are built — see Status above._
