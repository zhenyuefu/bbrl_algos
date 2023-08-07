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

import ffmpeg


def write_video_ffmpeg(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    fps: int = 30,
    out_fps: int = 30,
    vcodec: str = "libx264",
    input_fmt: str = "rgb24",
    output_fmt: str = "yuv420p",
) -> None:
    """Function that writes an video from a stream of numpy arrays using FFMPEG.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        out_fps: Frames per second for the saved video.
        vcodec: The video codec to use for the output video
        input_fmt: The input image format
        output_fmt: The output image format
    """

    first_img = next(itr)
    height, width, _ = first_img.shape

    stream = ffmpeg.input(
        "pipe:", format="rawvideo", pix_fmt=input_fmt, s=f"{width}x{height}", r=fps
    )
    stream = ffmpeg.output(
        stream, str(out_file), pix_fmt=output_fmt, vcodec=vcodec, r=out_fps
    )
    stream = ffmpeg.overwrite_output(stream)
    stream = ffmpeg.run_async(stream, pipe_stdin=True)

    def write_frame(img: np.ndarray) -> None:
        stream.stdin.write(as_uint8(img).tobytes())

    # Writes all the video frames to the file.
    write_frame(first_img)
    for img in itr:
        write_frame(img)

    stream.stdin.close()
    stream.wait()
