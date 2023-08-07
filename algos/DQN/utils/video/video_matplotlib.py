#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from typing import Iterator

import numpy as np

from . import as_uint8

import matplotlib.animation as ani
import matplotlib.pyplot as plt


def write_video_matplotlib(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    dpi: int = 50,
    fps: int = 30,
    title: str = "Video",
    comment: str | None = None,
    writer: str = "ffmpeg",
) -> None:
    """Function that writes an video from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib video writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    first_img = next(itr)
    height, width, _ = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Ensures that there is no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    writer_obj = ani.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = writer_obj(
        fps=fps,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(as_uint8(first_img), interpolation="nearest")
        mpl_writer.grab_frame()

        for img in itr:
            im.set_data(as_uint8(img))
            mpl_writer.grab_frame()
