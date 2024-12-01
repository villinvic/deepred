from typing import Tuple, Any

import numpy as np
from gymnasium import Space
from ml_collections import ConfigDict
from polaris.experience import SampleBatch
from polaris.models import BaseModel
from polaris.models.utils import CategoricalDistribution

from deepred.models.modules import Conv2DResidualModule, CategoricalValueHead

import tensorflow as tf
import sonnet as snt

class ImpalaShallowModel(BaseModel):
    """
    Non recurrent version of the resnet proposed in
    https://arxiv.org/pdf/1802.01561
    https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/experiment.py#L211
    We additionally accept additional ram information.
    TODO: We should use a categorical value function.
    """
    is_recurrent = False

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super().__init__(
            name="ImpalaShallowModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

        self.action_dist = CategoricalDistribution
        self.optimiser = snt.optimizers.RMSProp(
            learning_rate=config.lr,
            epsilon=1e-5,
            decay=0.99,
            momentum=0.,
        )

        self.conv_activation = config.get("conv_activation", tf.nn.relu)

        self.conv2D_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding='SAME')
            for num_ch, kernel_size, stride in [(8, 8, 4), (16, 4, 2)]
        ]

        ram_embedding_dims = config.get("ram_embedding_dims", [256])
        final_embedding_dims = config.get("final_embedding_dims", [256])

        self.ram_embedding = snt.nets.MLP(ram_embedding_dims, activate_final=True)
        self.final_embedding = snt.nets.MLP(final_embedding_dims, activate_final=True)

        self.policy_head = snt.Linear(self.action_space.n)
        self._value_logits = None
        self.value_head = snt.Linear(1)

        # num_value_bins = config.get("num_value_bins", [256])
        # value_bounds = config.get("value_bounds", (-5., 5.))
        # self.value_head = CategoricalValueHead(
        #     num_bins=num_value_bins,
        #
        # )


    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # The input is of shape (...), we need to add a Batch dimension.
        # we expand  once here as the pixels are in shape (w, h), should be (w, h, 1)
        pixels = tf.expand_dims(tf.expand_dims(tf.cast(obs["pixels"], tf.float32) / 255., axis=-1), axis=0)
        ram = tf.expand_dims(obs["ram"], axis=0)

        conv_out = pixels
        for conv in self.conv2D_layers:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        ram_embeddings = self.ram_embedding(ram)

        pixel_embeddings = snt.Flatten(1)(conv_out)
        prev_action_one_hot = tf.one_hot([prev_action], self.action_space.n, dtype=tf.float32)
        concat = tf.concat([pixel_embeddings, ram_embeddings, prev_action_one_hot], axis=-1)
        return self.final_embedding(concat)

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # The input is of shape (Time, Batch, ...)
        # we expand  once here as the pixels are in shape (w, h), should be (w, h, 1)
        pixels = tf.expand_dims(tf.cast(obs["pixels"], tf.float32) / 255., axis=-1)
        ram = obs["ram"]

        # We combine the Time and Batch dimensions for convolution
        shape = tf.shape(pixels)
        pixels = tf.reshape(pixels, tf.concat([[-1], shape[2:]], axis=0))
        conv_out = pixels
        for conv in self.conv2D_layers:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        ram_embeddings = self.ram_embedding(ram)

        pixel_embeddings = tf.reshape(conv_out, tf.concat([shape[:2], [-1]], axis=0))

        prev_action_one_hot = tf.one_hot(prev_action, self.action_space.n, dtype=tf.float32)
        concat = tf.concat([pixel_embeddings, ram_embeddings, prev_action_one_hot], axis=-1)
        return self.final_embedding(concat)

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        policy_logits = self.policy_head(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self.value_head(final_embeddings))
        }
        return policy_logits, state, extras

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        return self.policy_head(final_embeddings), state

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ) -> Tuple[Any, Any]:
        final_embeddings = self.batch_input(
            obs,
            prev_action,
            prev_reward,
            state
        )
        policy_logits = self.policy_head(final_embeddings)
        self._values = self.value_head(final_embeddings)[: ,:, 0]
        return policy_logits, self._values

    def critic_loss(
            self,
            vf_targets
    ):
        return tf.math.square(vf_targets-self._values)

    def get_initial_state(self):
        return (np.zeros(2, dtype=np.float32),)

    def get_metrics(self):
        return {}

