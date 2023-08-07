#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#


def compute_image_input_shape(
    input_shape: int | tuple[int, int] | tuple[int, int, int]
) -> tuple[tuple[int, int, int], int | None]:
    channel_pos: int | None = None
    if isinstance(input_shape, int):
        calc_input_shape = (1, input_shape, input_shape)
    elif isinstance(input_shape, tuple):
        if len(input_shape) == 1:
            calc_input_shape = (1, input_shape[0], input_shape[0])
        elif len(input_shape) == 2:
            calc_input_shape = (1,) + input_shape
        elif len(input_shape) == 3:
            # supposition: fewer channels than width or height
            if input_shape[0] < input_shape[1] and input_shape[0] < input_shape[2]:
                calc_input_shape, channel_pos = input_shape, 0
            else:
                calc_input_shape, channel_pos = (input_shape[2],) + input_shape[:-1], 2
        elif len(input_shape) == 4:
            raise ValueError("If input shape is 4D, consider using a custom Conv3d")
        else:
            raise ValueError(
                "Input shape must be int, 2D or 3D, got {}".format(input_shape)
            )
    else:
        raise TypeError("Input shape must be int or tuple")

    return calc_input_shape, channel_pos
