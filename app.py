import time

import anthropic
import cv2
import numpy as np
import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

from imaging import extract_palette, posterize

pillow_heif.register_heif_opener()

st.title("Underpainting")
st.write("Upload a photo and see how a painter would block it in.")

MAX_DIMENSION = 1200

# Hardcoded on purpose. Cut list item 1: no tunable UI, even if on schedule.
# A curated pair of studies is a better demo than a slider that lets a visitor
# find the ugliest output the app can produce.
COARSE_LEVELS = 3
FINE_LEVELS = 5

uploaded_file = st.file_uploader(
    "Upload a photo", type=["jpg", "jpeg", "png", "heic", "heif"]
)

if uploaded_file is not None:
    start = time.time()
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't read that file as an image. Try a JPEG, PNG, or HEIC photo.")
    else:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

        rgb_array = np.array(image)
        gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        coarse_study = posterize(gray_array, COARSE_LEVELS)
        fine_study = posterize(gray_array, FINE_LEVELS)
        palette = extract_palette(rgb_array)

        elapsed = time.time() - start

        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_array, caption="Original")
        with col2:
            st.image(gray_array, caption="Grayscale")

        st.caption(
            f"{rgb_array.shape[1]}×{rgb_array.shape[0]} px "
            f"· processed in {elapsed:.2f}s"
        )

        st.markdown("**Value studies**")

        col3, col4 = st.columns(2)
        with col3:
            st.image(coarse_study, caption=f"{COARSE_LEVELS}-value study")
        with col4:
            st.image(fine_study, caption=f"{FINE_LEVELS}-value study")

        # Verification readout, not polish. Shows the actual values present in
        # each study on this photo. Fewer than N means the photo has no pixels
        # in that band, which is information about the photo, not a bug.
        # Remove in S10.
        st.caption(
            f"{COARSE_LEVELS}-value: {np.unique(coarse_study).tolist()} · "
            f"{FINE_LEVELS}-value: {np.unique(fine_study).tolist()}"
        )

        st.markdown("**Palette**")
        st.caption(
            "Clustered in Lab so the groupings match colors a painter would mix "
            "as one. Ordered by how much of the canvas each covers."
        )

        for column, swatch in zip(st.columns(len(palette)), palette):
            with column:
                st.image(np.full((90, 200, 3), swatch["rgb"], dtype=np.uint8))
                st.caption(f"{swatch['hex']}\n\n{swatch['share'] * 100:.1f}%")

st.divider()
st.subheader("Model connection test")

if st.button("Ask Claude something"):
    with st.spinner("Asking Claude..."):
        try:
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=200,
                thinking={"type": "disabled"},
                messages=[{
                    "role": "user",
                    "content": "In one sentence, what is underpainting in traditional oil painting?",
                }],
            )
            reply = next(b.text for b in response.content if b.type == "text")
            st.write(reply)
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Check the secret in Streamlit Cloud settings.")
        except anthropic.APIStatusError as e:
            st.error(f"API error: {e.message}")
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the API. Check your internet connection.")
