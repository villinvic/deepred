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

from deepred.models.tf_utils import AdaptiveMaxPooling2D
from deepred.polaris_env.pokemon_red.enums import ProgressionFlag, BagItem, Move, Map
from deepred.polaris_env.pokemon_red.map_warps import NamedWarpIds


class SimpleModel(BaseModel):
    """
    From
    https://github.com/CJBoey/PokemonRedExperiments1/blob/master/baselines/boey_baselines2/custom_network.py
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
            name="SimpleModel",
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


        self.screen_conv_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding=padding, name=f"screen{kernel_size}")
            for num_ch, kernel_size, stride, padding in [(32, 8, 4, "VALID"), (64, 4, 2, "VALID"), (64, 3, 1, "VALID")]
        ]
        self.screen_conv_max_pool = tf.keras.layers.MaxPooling2D((3, 1), strides=(3, 1))

        self.map_features_conv_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding=padding, name=f"map{kernel_size}")
            for num_ch, kernel_size, stride, padding in [(32, 4, 1, "VALID"), (64, 4, 1, "VALID"), (64, 3, 1, "VALID")]
        ]

        self.screen_embedding = snt.nets.MLP([128], activate_final=True, name=f"screen_embed")
        self.map_features_embedding = snt.nets.MLP([128], activate_final=True, name=f"screen_embed")

        self.maps_mlp = snt.nets.MLP([32, 32], activate_final=True, name=f"maps_mlp")
        self.maps_max_pool = AdaptiveMaxPooling2D((1, 32), channels_last=False)

        self.sprites_embedding = snt.Embed(390, 8, densify_gradients=True, name=f"sprites_embedding")
        self.warps_embedding = snt.Embed(len(NamedWarpIds), 8, densify_gradients=True, name=f"warps_embedding")
        self.maps_embedding = snt.Embed(len(Map) + 1, 32, densify_gradients=True, name="maps_embedding")

        self.final_mlp = snt.nets.MLP([32,32], activate_final=True, name="final_mlp")
        self.policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self._value_logits = None
        self.value_head = snt.Linear(1, name="value_head")

        # num_value_bins = config.get("num_value_bins", [256])
        # value_bounds = config.get("value_bounds", (-5., 5.))
        # self.value_head = CategoricalValueHead(
        #     num_bins=num_value_bins,
        #
        # )

        # TODO: Care
        #       - add mask > 0 for pps


    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # The input is of shape (...), we need to add a Batch dimension.
        # we expand  once here as the pixels are in shape (w, h), should be (w, h, 1)
        pixels = tf.expand_dims(tf.expand_dims(tf.cast(obs["main_screen"], tf.float32) / 255., axis=-1), axis=0)

        conv_out = self.screen_conv_layers[0](pixels)
        conv_out = self.conv_activation(conv_out)
        conv_out = self.screen_conv_max_pool(conv_out)
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        screen_embed = self.screen_embedding(conv_out_flat)

        feature_maps = tf.transpose(tf.expand_dims(tf.cast(obs["feature_maps"], tf.float32) / 255., axis=0), perm=[0, 2, 3, 1])
        sprite_map = tf.expand_dims(self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64)), axis=0)
        warp_map = tf.expand_dims(self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64)), axis=0)
        feature_maps = tf.concat([feature_maps, sprite_map, warp_map], axis=-1)

        conv_out = self.map_features_conv_layers[0](feature_maps)
        conv_out = self.conv_activation(conv_out)
        for conv in self.map_features_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        map_features_embed = self.map_features_embedding(conv_out_flat)

        maps = tf.expand_dims(self.maps_embedding(tf.cast(obs['map_ids'], tf.int64)), axis=0)
        maps_embed = tf.reduce_max(maps, axis=-2)

        additional_ram_info = tf.expand_dims(obs["ram"], axis=0)


        concat = tf.concat([
            additional_ram_info,
            # screen_embed,
            # maps_embed,
            # map_features_embed
        ], axis=-1)
        return self.final_mlp(concat)

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # The input is of shape (Time, Batch, ...)
        pixels = tf.expand_dims(tf.cast(obs["main_screen"], tf.float32) / 255., axis=-1)

        shape = tf.shape(pixels)
        pixels = tf.reshape(pixels, tf.concat([[-1], shape[2:]], axis=0))
        conv_out = self.screen_conv_layers[0](pixels)
        conv_out = self.conv_activation(conv_out)
        conv_out = self.screen_conv_max_pool(conv_out)
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        screen_embed = self.screen_embedding(conv_out_flat)
        screen_embed = tf.reshape(screen_embed, tf.concat([shape[:2], [-1]], axis=0))


        feature_maps = tf.transpose(tf.cast(obs["feature_maps"], tf.float32) / 255., perm=[0, 1, 3, 4, 2])
        sprite_map = self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64))
        warp_map = self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64))
        feature_maps = tf.concat([feature_maps, sprite_map, warp_map], axis=-1)

        shape = tf.shape(feature_maps)
        feature_maps = tf.reshape(feature_maps, tf.concat([[-1], shape[2:]], axis=0))
        conv_out = self.map_features_conv_layers[0](feature_maps)
        conv_out = self.conv_activation(conv_out)
        for conv in self.map_features_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        map_features_embed = self.map_features_embedding(conv_out_flat)
        map_features_embed = tf.reshape(map_features_embed, tf.concat([shape[:2], [-1]], axis=0))

        maps = self.maps_embedding(tf.cast(obs['map_ids'], tf.int64))
        maps_embed = tf.reduce_max(maps, axis=-2)
        #map_features = self.map_ids_fc_relu(map_concat)  # (20, 16)
        #map_features = self.map_ids_max_pool(map_features).squeeze(-2)  # (20, 16) -> (16, )

        additional_ram_info = obs["ram"]

        concat = tf.concat([
                            additional_ram_info,
                            # screen_embed,
                            # maps_embed,
                            # map_features_embed
                            ], axis=-1)
        return self.final_mlp(concat)


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

