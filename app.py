import io
import time

import cv2
import numpy as np
import PIL
import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

st.title("Underpainting")
st.write("Toolchain works. Session 1 done.")

st.subheader("Library versions")
st.write(f"numpy: {np.__version__}")
st.write(f"opencv-python-headless: {cv2.__version__}")
st.write(f"pillow: {PIL.__version__}")

st.subheader("Animated GIF test")

frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
frame_a[:, :] = (30, 120, 200)

frame_b = cv2.bitwise_not(frame_a)

frames = [Image.fromarray(frame_a), Image.fromarray(frame_b)]
buffer = io.BytesIO()
frames[0].save(
    buffer,
    format="GIF",
    save_all=True,
    append_images=frames[1:],
    duration=500,
    loop=0,
)
buffer.seek(0)

st.image(buffer.getvalue())

st.divider()
st.subheader("Upload a photo")

MAX_DIMENSION = 1200

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
