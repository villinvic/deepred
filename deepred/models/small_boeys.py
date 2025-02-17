import math
from typing import Tuple, Any

import numpy as np
from gymnasium import Space
from ml_collections import ConfigDict
from polaris.experience import SampleBatch
from polaris.models import BaseModel
from polaris.models.utils import CategoricalDistribution
from sonnet import initializers

from deepred.models.modules import Conv2DResidualModule, CategoricalValueHead, ContextualAttentionPooling

import tensorflow as tf
import sonnet as snt

from deepred.polaris_env.pokemon_red.enums import ProgressionFlag, BagItem, Move, Map
from deepred.polaris_env.pokemon_red.map_warps import NamedWarpIds


class SmallBoeysModel(BaseModel):
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
            name="SmallBoeysModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

        self.action_dist = CategoricalDistribution
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )
        # self.optimiser = snt.optimizers.RMSProp(
        #     learning_rate=config.lr,
        #     epsilon=1e-5,
        #     decay=0.99,
        #     momentum=0.,
        # )

        self.conv_activation = config.get("conv_activation", tf.nn.relu)

        self.screen_conv_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding=padding, name=f"screen{kernel_size}"
                       )
            for num_ch, kernel_size, stride, padding in [(32, 8, 2, "VALID"), (64, 4, 2, "VALID"), (64, 3, 2, "VALID")]
        ]

        self.map_features_conv_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding=padding, name=f"map{kernel_size}")
            for num_ch, kernel_size, stride, padding in [(32, 4, 1, "VALID"), (64, 3, 1, "VALID"), (64, 2, 1, "VALID")]
        ]


        self.screen_embedding = snt.nets.MLP([512], activate_final=True, name="screen_embedding")
        self.map_features_embedding = snt.nets.MLP([512], activate_final=True, name="map_features_embedding")

        self.move_mlp = snt.nets.MLP([32, 32], activate_final=True, name="move_mlp")
        self.context_mlp = snt.nets.MLP([32], activate_final=True, name="context_mlp")

        self.pokemon_mlp = snt.nets.MLP([64], activate_final=True, name="pokemon_mlp")

        self.party_attention = ContextualAttentionPooling(embed_dim=64)

        self.move_attention = ContextualAttentionPooling(embed_dim=32)


        self.items_mlp = snt.nets.MLP([8, 8], activate_final=True, name="items_mlp")
        self.events_mlp = snt.nets.MLP([32, 32], activate_final=True, name="events_mlp")

        self.sprites_embedding = snt.Embed(390, 8, densify_gradients=True, name="sprite_embedding")
        self.warps_embedding = snt.Embed(len(NamedWarpIds)+1, 8, densify_gradients=True, name="warps_embedding")
        self.moves_embedding = snt.Embed(len(Move) + 1, 16, densify_gradients=True, name="moves_embedding")
        self.types_embedding = snt.Embed(17, 8, densify_gradients=True, name="types_embedding")
        # no pokemon id embedding
        self.items_embedding = snt.Embed(256, 8, densify_gradients=True, name="bag_items_embedding")
        self.events_embedding = snt.Embed(len(ProgressionFlag)+1, 32, densify_gradients=True, name="event_embedding")
        self.maps_embedding = snt.Embed(len(Map)+1, 32, densify_gradients=True, name="maps_embedding")

        self.final_mlp = snt.nets.MLP([512, 512], activate_final=True, name="final_mlp")
        self.policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self.value_head = snt.Linear(1, name="value_head")

        # num_value_bins = config.get("num_value_bins", 75)
        # value_bounds = config.get("value_bounds", (-10., 30.))
        # self.value_head = CategoricalValueHead(
        #     num_bins=num_value_bins,
        #     value_bounds=value_bounds,
        #     smoothing_ratio=0.75
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

        pixels = tf.expand_dims(tf.cast(obs["main_screen"], tf.float32) / 255., axis=-1)

        conv_out = self.screen_conv_layers[0](pixels)
        conv_out = self.conv_activation(conv_out)
        #conv_out = self.screen_conv_max_pool(conv_out)
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)

        conv_out_flat = snt.Flatten(1)(conv_out)
        screen_embed = self.screen_embedding(conv_out_flat)

        feature_maps = tf.transpose(tf.cast(obs["feature_maps"], tf.float32) / 255., perm=[0, 2, 3, 1])
        sprite_map = self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64))
        warp_map = self.sprites_embedding(tf.cast(obs["sprite_map"], tf.int64))
        feature_maps = tf.concat([feature_maps, sprite_map, warp_map], axis=-1)

        conv_out = self.map_features_conv_layers[0](feature_maps)
        conv_out = self.conv_activation(conv_out)
        for conv in self.map_features_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        map_features_embed = self.map_features_embedding(conv_out_flat)

        in_battle_mask = tf.convert_to_tensor(obs["is_in_battle"], dtype=tf.float32)

        # sent out party embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        move_index_one_hots = tf.expand_dims(tf.eye(4, dtype=tf.float32), axis=0)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        sent_out_indexed_moves_embed = tf.concat([moves_embed, move_index_one_hots], axis=-1)
        moves_embed = tf.reduce_max(moves_embed, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_party_attributes"]
        sent_out_party_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        # sent out opp embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_opp_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_opp_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)
        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_opp_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_opp_attributes"]
        sent_out_opp_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        sent_out_opp_info = sent_out_opp_info * in_battle_mask

        context = self.context_mlp(tf.concat([sent_out_party_info, sent_out_opp_info], axis=-1))


        attended_moves = self.move_attention(
            query=context,
            key_value=sent_out_indexed_moves_embed,
        )

        # party pokemon embedding
        moves = self.moves_embedding(tf.cast(obs["party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["party_attributes"]
        party_info = tf.concat([moves_embed, types, attributes], axis=-1)
        party_embed = self.pokemon_mlp(party_info)

        attended_party = self.party_attention(
            query=context,
            key_value=party_embed,
        )

        items = self.items_embedding(tf.cast(obs['item_ids'], tf.int64))
        item_quantities = tf.expand_dims(obs['item_quantities'], axis=-1)
        items_info = tf.concat([items, item_quantities], axis=-1)
        items_embed = self.items_mlp(items_info)
        items_embed = tf.reduce_max(items_embed, axis=-2)

        events = self.events_embedding(tf.cast(obs['recent_event_ids'], tf.int64))
        # events_age = tf.expand_dims(obs['recent_event_ids_age'], axis=-1)
        # events_info = tf.concat([events, events_age], axis=-1)
        # events_embed = self.events_mlp(events_info)
        events_embed = tf.reduce_max(events, axis=-2)

        maps = self.maps_embedding(tf.cast(obs['map_ids'], tf.int64))
        maps_embed = tf.reduce_max(maps, axis=-2)

        additional_ram_info = obs["ram"]

        concat = tf.concat([
            maps_embed,
            events_embed,
            map_features_embed,
            screen_embed,
            items_embed,
            attended_moves,
            context,
            attended_party,
            additional_ram_info,
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
        #conv_out = self.screen_conv_max_pool(conv_out)
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        conv_out = tf.reshape(conv_out_flat, tf.concat([shape[:2], [-1]], axis=0))
        screen_embed = self.screen_embedding(conv_out)


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

        in_battle_mask = tf.cast(obs["is_in_battle"], dtype=tf.float32)

        # sent out party embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        move_index_one_hots = tf.expand_dims(tf.eye(4, dtype=tf.float32), axis=0)
        move_index_one_hots = tf.tile(tf.expand_dims(move_index_one_hots, axis=0), tf.concat([shape[:2], [1, 1]], axis=0))

        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        sent_out_indexed_moves_embed = tf.concat([moves_embed, move_index_one_hots], axis=-1)
        moves_embed = tf.reduce_max(moves_embed, axis=-2)


        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_party_attributes"]
        sent_out_party_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        # sent out opp embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_opp_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_opp_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)
        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_opp_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_opp_attributes"]
        sent_out_opp_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        sent_out_opp_info = sent_out_opp_info * in_battle_mask

        context = self.context_mlp(tf.concat([sent_out_party_info, sent_out_opp_info], axis=-1))

        attended_moves = self.move_attention(
            query=context,
            key_value=sent_out_indexed_moves_embed,
        )

        # party pokemon embedding
        moves = self.moves_embedding(tf.cast(obs["party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["party_attributes"]
        party_info = tf.concat([moves_embed, types, attributes], axis=-1)
        party_embed = self.pokemon_mlp(party_info)

        attended_party = self.party_attention(
            query=context,
            key_value=party_embed,
        )

        items = self.items_embedding(tf.cast(obs['item_ids'], tf.int64))
        item_quantities = tf.expand_dims(obs['item_quantities'], axis=-1)
        items_info = tf.concat([items, item_quantities], axis=-1)
        items_embed = self.items_mlp(items_info)
        items_embed = tf.reduce_max(items_embed, axis=-2)

        events = self.events_embedding(tf.cast(obs['recent_event_ids'], tf.int64))
        # events_age = tf.expand_dims(obs['recent_event_ids_age'], axis=-1)
        # events_info = tf.concat([events, events_age], axis=-1)
        # events_embed = self.events_mlp(events_info)
        events_embed = tf.reduce_max(events, axis=-2)

        maps = self.maps_embedding(tf.cast(obs['map_ids'], tf.int64))
        maps_embed = tf.reduce_max(maps, axis=-2)

        additional_ram_info = obs["ram"]

        concat = tf.concat([
            maps_embed,
            events_embed,
            map_features_embed,
            screen_embed,
            items_embed,
            attended_moves,
            context,
            attended_party,
            additional_ram_info,
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
        self._values = self.value_head(final_embeddings)[:, :, 0]
        return policy_logits, self._values

    def critic_loss(
            self,
            vf_targets
    ):
        return tf.math.square(vf_targets-self._values)
        #return self.value_head.loss(vf_targets)

    def get_initial_state(self):
        return (np.zeros(2, dtype=np.float32),)

    def get_metrics(self):
        return {}

