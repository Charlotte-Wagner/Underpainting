"""GIF encoding for Underpainting's cross-fade build-order animation.

Kept out of imaging.py because the output here is GIF bytes through Pillow,
not a numpy array back out. Same standard as imaging.py otherwise: zero
Streamlit imports, runnable against arrays built by hand, no framework
dependency.
"""

from io import BytesIO

import numpy as np
from PIL import Image

STAGE_DURATION_MS = 650
FADE_DURATION_MS = 45
PALETTE_SAMPLE_ROWS = 512


def frame_durations(stage_count, fade_frames):
    """Per-frame duration in ms, matching imaging.cross_fade_frames's order.

    A stage frame holds long enough to actually read as a pause. A fade
    frame is short, since its whole job is to be in between. Holding via
    duration instead of duplicate frames keeps frame count, and therefore
    file size, down without shortening the pause on screen.
    """
    durations = [STAGE_DURATION_MS]
    for _ in range(stage_count - 1):
        durations.extend([FADE_DURATION_MS] * fade_frames)
        durations.append(STAGE_DURATION_MS)
    return durations


def _to_image(frame):
    return Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")


def _shared_palette(frames, colors=256):
    """One 256-color palette built from rows sampled across every frame.

    Pillow quantizes each frame to its own palette by default, so consecutive
    GIF frames end up with slightly different color tables and the animation
    flickers on loop. Sampling rows from every frame into one composite image
    and quantizing that once means every frame that gets matched against it
    afterward shares the identical table.

    The rows are spread evenly down each frame rather than taken as a strip
    off the top, which is a fix, not a preference. A top strip only represents
    the frame if the frame is uniform vertically, and S12's line drawing is
    not: the sample photo's drawing has nothing but flat toned ground in its
    top quarter, so pure black never reached the sample, never got a palette
    entry, and every line in the first frame came out rendered in the sky's
    dark blue. Same class of bug as sampling one photo to tune a constant.
    """
    rows_per_frame = max(1, PALETTE_SAMPLE_ROWS // len(frames))
    height = np.asarray(frames[0]).shape[0]
    rows = np.unique(np.linspace(0, height - 1, min(rows_per_frame, height)).astype(int))
    strips = [np.asarray(f, dtype=np.uint8)[rows] for f in frames]
    composite = Image.fromarray(np.concatenate(strips, axis=0), mode="RGB")
    return composite.quantize(colors=colors)


def encode_gif(frames, durations):
    """Encode frames as a looping GIF with one shared palette.

    Args:
        frames: list of uint8 RGB arrays, all the same shape.
        durations: list of int milliseconds, same length as frames.

    Returns:
        Raw GIF bytes.
    """
    if len(frames) != len(durations):
        raise ValueError(
            "frames and durations must be the same length, got %d and %d"
            % (len(frames), len(durations))
        )

    palette = _shared_palette(frames)
    quantized = [
        _to_image(frame).quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
        for frame in frames
    ]

    # Quantizing every frame against the same reference palette image makes
    # their pixel indices agree, but Pillow still writes a separate local
    # color table per frame unless the save-time `palette` argument is also
    # set: without it, GifImagePlugin defaults `optimize=True` and gives every
    # frame after the first its own color table, which is the flicker this
    # whole function exists to avoid. Passing the same table here is what
    # actually makes the frames share one global color table in the file.
    buffer = BytesIO()
    quantized[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        palette=bytes(palette.getpalette()),
        optimize=False,
    )
    return buffer.getvalue()
