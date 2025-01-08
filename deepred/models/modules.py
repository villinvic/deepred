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

        return conv_out


class ContextualAttentionPooling(snt.Module):
    def __init__(
            self,
            embed_dim: int,
            name: str = "ContextualAttentionPooling"
    ):
        """
        Contextual Attention Pooling module for embeddings.

        :param embed_dim: The dimensionality of the Pokémon embeddings.
        :param name: Optional name for the module.
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.key_layer = snt.Linear(embed_dim)  # For keys
        self.query_layer = snt.Linear(embed_dim)  # For queries
        self.value_layer = snt.Linear(embed_dim)  # For values


    def __call__(
            self,
            query: tf.Tensor,
            key: tf.Tensor,
            value: tf.Tensor,
            preprocessed_value: False,
            indexed: bool = True,

    ) -> tf.Tensor:
        """
        Perform attention pooling with context.
        """

        if query.shape != key.shape:
            query = tf.expand_dims(query, axis=-2)

        query = self.query_layer(query)
        key = self.key_layer(key)
        if preprocessed_value:
            value = value
        else:
            value = self.value_layer(value)

        attn_output = tf.matmul(query, key, transpose_b=True)  # Query x Key ^T
        attn_weights = tf.nn.softmax(attn_output, axis=-1)  # Normalize attention weights with softmax

        shape = tf.shape(value)
        one_hots = tf.expand_dims(tf.eye(shape[-2], dtype=tf.float32), axis=0)
        if len(shape) > 3:
            one_hots = tf.tile(tf.expand_dims(one_hots, axis=0), tf.concat([shape[:-2], [1, 1]], axis=0))
        indexed_values = tf.concat([value, one_hots], axis=-1)

        attended_embedding = tf.matmul(attn_weights, indexed_values)  # Shape: (batch_size, 1, embed_dim)
        attended_embedding = tf.reduce_sum(attended_embedding, axis=-2)  # Shape: (batch_size, embed_dim)

        return attended_embedding


class ContextualMultiHeadedAttentionPooling(snt.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, name: str = "ContextualMultiHeadedAttentionPooling"):
        """
        Contextual Attention Pooling module with multi-head attention.

        :param embed_dim: Total dimensionality of the embeddings.
        :param num_heads: Number of attention heads.
        :param name: Optional name for the module.
        """
        super().__init__(name=name)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate linear layers for key, query, and value projections
        self.key_layer = snt.Linear(embed_dim)
        self.query_layer = snt.Linear(embed_dim)
        self.value_layer = snt.Linear(embed_dim)

        # Final projection layer to combine multi-head outputs
        self.output_layer = snt.Linear(embed_dim)

    def __call__(self, query: tf.Tensor, key_value: tf.Tensor) -> tf.Tensor:
        """
        Perform multi-head attention pooling with context.

        :param query: Query tensor, shape (batch_size, embed_dim).
        :param key_value: Key/Value tensor, shape (batch_size, seq_len, embed_dim).
        :return: Attended embedding, shape (batch_size, embed_dim).
        """
        T, B = tf.shape(query)[0], tf.shape(query)[1]
        seq_len = tf.shape(key_value)[-2]

        # Project query, key, and value to multiple heads
        query = self.query_layer(query)  # Shape: (batch_size, embed_dim)
        key = self.key_layer(key_value)  # Shape: (batch_size, seq_len, embed_dim)
        value = self.value_layer(key_value)  # Shape: (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention
        query = tf.reshape(query,(T, B, 1, self.num_heads, self.head_dim))
        key = tf.reshape(key, (T, B, seq_len, self.num_heads, self.head_dim))
        value = tf.reshape(value, (T, B, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch_size, num_heads, seq_len, head_dim) for key and value
        query = tf.transpose(query, perm=[0, 2, 1, 3])  # (batch_size, num_heads, 1, head_dim)
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, head_dim)
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        attn_logits = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, 1, seq_len)
        attn_logits = attn_logits / tf.sqrt(float(self.head_dim))  # Scale by sqrt(head_dim)
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)  # Normalize weights over seq_len

        # Compute attended values
        attended_heads = tf.matmul(attn_weights, value)  # (batch_size, num_heads, 1, head_dim)

        # Concatenate heads and project back to embed_dim
        attended_heads = tf.transpose(attended_heads, perm=[0, 2, 1, 3])  # (batch_size, 1, num_heads, head_dim)
        attended_heads = tf.reshape(attended_heads, (batch_size, -1, self.embed_dim))  # (batch_size, 1, embed_dim)

        attended_embedding = self.output_layer(tf.squeeze(attended_heads, axis=-2))  # (batch_size, embed_dim)

        return attended_embedding



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
            (self.support - tf.expand_dims(targets, axis=-1))
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





if __name__ == '__main__':
    # Hyperparameters
    time_size = 8
    batch_size = 4
    seq_len = 6  # Number of items in the sequence (e.g., Pokémon party or move pool)
    embed_dim = 16  # Embedding dimensionality

    # Initialize the module
    contextual_pooling = ContextualAttentionPooling(embed_dim=embed_dim)

    # Create dummy query (e.g., sent-out Pokémon context)
    query = tf.random.normal(shape=(time_size, batch_size, embed_dim))

    # Create dummy key-value sequence (e.g., party Pokémon embeddings)
    key_value = tf.random.normal(shape=(time_size, batch_size, seq_len, embed_dim))

    # Apply the contextual attention pooling
    pooled_embedding = contextual_pooling(query, key_value)

    # Print results
    print("Query shape:", query.shape)  # (batch_size, embed_dim)
    print("Key-value shape:", key_value.shape)  # (batch_size, seq_len, embed_dim)
    print("Pooled embedding shape:", pooled_embedding.shape)  # (batch_size, embed_dim)

    # Assertions to validate the output
    assert pooled_embedding.shape == (time_size, batch_size, embed_dim), "Output shape is incorrect."
    print("Test passed: Output shape is as expected.")
