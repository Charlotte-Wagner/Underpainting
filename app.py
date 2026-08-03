import io

import cv2
import numpy as np
import PIL
import streamlit as st
from PIL import Image

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
