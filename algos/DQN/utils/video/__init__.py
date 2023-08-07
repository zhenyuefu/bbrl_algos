#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#
import shutil
import warnings
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np


Writer = Literal["ffmpeg", "matplotlib", "opencv"]
WRITERS: dict[Writer, Callable[[Iterator[np.ndarray], str | Path], None]] = {}


def as_uint8(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr * 255).round().astype(np.uint8)
    raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")


try:
    from .video_ffmpeg import write_video_ffmpeg

    assert shutil.which("ffmpeg"), "FFmpeg not found in PATH"
    WRITERS["ffmpeg"] = write_video_ffmpeg
except ModuleNotFoundError | AssertionError:
    warnings.warn("FFmpeg not found, its writer will not be available.")

try:
    from .video_matplotlib import write_video_matplotlib

    WRITERS["matplotlib"] = write_video_matplotlib
    if not shutil.which("ffmpeg"):
        warnings.warn(
            "FFmpeg not found, the default matplotlib ani writer will not be available."
        )
except ModuleNotFoundError:
    warnings.warn("Matplotlib not found, its writer will not be available.")

try:
    from .video_opencv import write_video_opencv

    WRITERS["opencv"] = write_video_opencv
except ModuleNotFoundError:
    warnings.warn("OpenCV not found, its writer will not be available.")
