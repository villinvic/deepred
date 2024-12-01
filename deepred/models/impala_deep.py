from typing import Tuple, Any

from gymnasium import Space
from ml_collections import ConfigDict
from polaris.experience import SampleBatch
from polaris.models import BaseModel
from polaris.models.utils import CategoricalDistribution

from deepred.models.modules import Conv2DResidualModule, CategoricalValueHead

import tensorflow as tf
import sonnet as snt

class ImpalaDeepModel(BaseModel):
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
            name="ImpalaDeepModel",
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


        num_channels = config.get("num_channels", [16, 32, 32])
        num_blocks = config.get("num_channels", 2)
        kernel_shape = config.get("kernel_shape", (3, 3))
        stride = config.get("stride", 1)
        pooling_stride = config.get("pooling_stride", (2, 2))
        pooling_window = config.get("pooling_window", (3, 3))
        self.conv_activation = config.get("conv_activation", tf.nn.relu)

        self.residual_blocks = [
            Conv2DResidualModule(
                num_channels=num_ch,
                num_blocks=num_blocks,
                kernel_shape=kernel_shape,
                stride=stride,
                pooling_stride=pooling_stride,
                pooling_window=pooling_window,
                activation=self.conv_activation,
            )
            for num_ch in num_channels
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


    def _torso(
            self,
            obs,
    ):
        pixels = tf.cast(obs["pixels"], tf.float32) / 255.
        ram = obs["ram"]

        conv_out = pixels
        for residual_block in self.residual_blocks:
            conv_out = residual_block(conv_out)
        conv_out = self.conv_activation(conv_out)
        ram_embeddings = self.ram_embedding(ram)
        return conv_out, ram_embeddings


    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        conv_out, ram_embeddings = self._torso(obs)
        pixel_embeddings = snt.Flatten(1)(conv_out)
        prev_action_one_hot = tf.one_hot(prev_action, self.action_space.n, dtype=tf.float32)
        concat = tf.concat([pixel_embeddings, ram_embeddings, prev_action_one_hot], axis=-1)
        return self.final_embedding(concat)

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        conv_out, ram_embeddings = self._torso(obs)
        pixel_embeddings = snt.Flatten(2)(conv_out)
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
        self._value_logits = tf.squeeze(self.value_head(final_embeddings))
        return policy_logits, self._value_logits

    def critic_loss(
            self,
            vf_targets
    ):
        return tf.math.square(vf_targets-self._value_logits)


