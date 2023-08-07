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

import cv2


def write_video_opencv(
    itr: Iterator[np.ndarray],
    out_file: str | Path,
    fps: int = 30,
    codec: str = "mp4v",
) -> None:
    """Function that writes a video from a stream of numpy arrays using OpenCV.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        fps: Frames per second for the video.
        codec: FourCC code specifying OpenCV video codec type. Examples are
            MPEG, MP4V, DIVX, AVC1, H236.
    """

    first_img = next(itr)
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    stream = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    def write_frame(img: np.ndarray) -> None:
        stream.write(as_uint8(img))

    write_frame(first_img)
    for img in itr:
        write_frame(img)

    stream.release()
    cv2.destroyAllWindows()
