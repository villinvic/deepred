from typing import Union, Iterable
import sonnet as snt
import tensorflow as tf


def normalize_data_format(value):
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            '"channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format

def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    A copy of tensorflow.python.keras.util.

    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple

class AdaptiveMaxPooling2D(snt.Module):
    """
    TF alternative of the pytorch's AdaptiveMaxPool2d

    Max Pooling with adaptive kernel size.

    Args:
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

    """

    def __init__(
        self,
        output_size: Union[int, Iterable[int]],
        channels_last: bool = True,
        **kwargs,
    ):
        self.channels_last = channels_last
        self.output_size = normalize_tuple(output_size, 2, "output_size")
        super().__init__(**kwargs)

        if self.channels_last:
            self.data_format = "channels_last" #"NHWC"
            self.input_shape_getter = lambda inputs:  tf.shape(inputs)[-3:-1]
        else:
            self.data_format = "channels_first"  #"NCHW"
            self.input_shape_getter = lambda inputs:  tf.shape(inputs)[-2:]


    @snt.once
    def initialise(self, inputs):
        shape = self.input_shape_getter(inputs)

        stride1 = (shape[0] // self.output_size[0])
        stride2 = (shape[1] // self.output_size[1])

        kernel_size1 = shape[0] - (self.output_size[0] - 1) * stride1
        kernel_size2 = shape[1] - (self.output_size[1] - 1) * stride2

        self.maxpool_op = tf.keras.layers.MaxPool2D(
            pool_size=(kernel_size1, kernel_size2),
            strides=(stride1, stride2),
            padding="valid",
            data_format=self.data_format
        )


    def __call__(self, inputs, *args):
        self.initialise(inputs)
        return self.maxpool_op(inputs)
