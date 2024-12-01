from typing import Union, Tuple, Callable
import sonnet as snt
import tensorflow as tf


class Conv2DResidualModule(snt.Module):
    # From https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/experiment.py#L211

    def __init__(
            self,
            num_channels: int = 16,
            num_blocks: int = 2,
            kernel_shape: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            pooling_stride: Tuple[int, int] = (2, 2),
            pooling_window: Tuple[int, int] = (3, 3),
            activation: Callable = tf.nn.relu
    ):
        super().__init__(name="Conv2DResidualModule")
        self.num_blocks = num_blocks
        self.pooling_stride = pooling_stride
        self.pooling_window = pooling_window
        self.activation = activation
        self.conv_layers = [snt.Conv2D(num_channels, kernel_shape, stride=stride, padding='SAME') for _ in
                            range(1 + 2*self.num_blocks)]

    def __call__(self, residual):
        # Downscale.
        conv_out = self.conv_layers[0](residual)
        conv_out = tf.nn.pool(
            conv_out,
            window_shape=self.pooling_window,
            pooling_type='MAX',
            padding='SAME',
            strides=self.pooling_stride)

        for j in range(self.num_blocks):
            block_input = conv_out
            conv_out = self.activation(conv_out)
            conv_out = self.conv_layers[j*2+1](conv_out)
            conv_out = self.activation(conv_out)
            conv_out = self.conv_layers[j*2+2](conv_out)
            conv_out += block_input

        return residual


class CategoricalValueHead(snt.Module):
    # https://arxiv.org/pdf/2403.03950

    def __init__(
            self,
            num_bins=50,
            value_bounds=(-5., 5.),
            smoothing_ratio=0.75,
    ):
        super().__init__(name='CategoricalValueHead')
        self.num_bins = num_bins
        self.value_bounds = value_bounds
        self.bin_width = (self.value_bounds[1] - self.value_bounds[0]) / self.num_bins
        self.support = tf.cast(tf.expand_dims(tf.expand_dims(tf.linspace(*self.value_bounds, self.num_bins + 1), axis=0), axis=0),
                               tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma

        self.value_out = snt.Linear(self.num_bins)
        self._logits = None

    def __call__(
            self,
            input_
    ):
        self._logits = self.value_out(input_)
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._logits),
                              axis=-1)

    def targets_to_probs(self, targets):

        # this may occur on rare occasion that targets are outside of the set interval.
        targets = tf.clip_by_value(targets, *self.value_bounds)

        cdf_evals = tf.math.erf(
            (self.support - tf.expand_dims(targets, axis=2))
            / self.sqrt_two_sigma
        )
        z = cdf_evals[:, :, -1:] - cdf_evals[:, :,  :1]
        bin_probs = cdf_evals[:, :, 1:] - cdf_evals[:, :, :-1]
        ret = bin_probs / z

        return ret

    def loss(self, targets):

        # HL-Gauss classification loss
        return tf.losses.categorical_crossentropy(
            y_true=self.targets_to_probs(targets),
            y_pred=self._logits,
            from_logits=True,
        )