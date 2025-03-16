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


class ArenaModel(BaseModel):  # base is small_boeys.py
    """
    From
    https://github.com/CJBoey/PokemonRedExperiments1/blob/master/baselines/boey_baselines2/custom_network.py
    """
    is_recurrent = False

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super().__init__(
            name="ArenaModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

        self.action_dist = CategoricalDistribution
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )

        self.conv_activation = config.get("conv_activation", tf.nn.relu)

        self.screen_conv_layers = [
            snt.Conv2D(num_ch, kernel_size, stride=stride, padding=padding, name=f"screen{kernel_size}"
                       )
            for num_ch, kernel_size, stride, padding in [(32, 8, 2, "VALID"), (64, 4, 2, "VALID"), (64, 3, 2, "VALID")]
        ]

        self.screen_embedding = snt.nets.MLP([512], activate_final=True, name="screen_embedding")

        self.move_mlp = snt.nets.MLP([32, 32], activate_final=True, name="move_mlp")
        self.context_mlp = snt.nets.MLP([32], activate_final=True, name="context_mlp")

        self.pokemon_mlp = snt.nets.MLP([64], activate_final=True, name="pokemon_mlp")

        self.party_attention = ContextualAttentionPooling(embed_dim=64)

        self.move_attention = ContextualAttentionPooling(embed_dim=32)

        self.items_mlp = snt.nets.MLP([8, 8], activate_final=True, name="items_mlp")
        self.moves_embedding = snt.Embed(len(Move) + 1, 16, densify_gradients=True, name="moves_embedding")
        self.types_embedding = snt.Embed(17, 8, densify_gradients=True, name="types_embedding")
        # no pokemon id embedding
        self.items_embedding = snt.Embed(256, 8, densify_gradients=True, name="bag_items_embedding")

        self.final_mlp = snt.nets.MLP([512, 512], activate_final=True, name="final_mlp")
        self.policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self.value_head = snt.Linear(1, name="value_head")

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
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)

        conv_out_flat = snt.Flatten(1)(conv_out)
        screen_embed = self.screen_embedding(conv_out_flat)

        in_battle_mask = tf.convert_to_tensor(obs["is_in_battle"], dtype=tf.float32)

        # sent out party embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_info = tf.concat([moves, pps, pps_mask], axis=-1)
        moves_embed_pre_pool = self.move_mlp(moves_info)
        moves_embed = tf.reduce_max(moves_embed_pre_pool, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_party_attributes"]

        sent_out_index = tf.one_hot(tf.cast(obs["sent_out_party_index"], tf.int64), 6, dtype=tf.float32)[:, 0]
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

        battle_context = tf.concat([sent_out_party_info, sent_out_opp_info], axis=-1)

        attended_moves = self.move_attention(
            query=battle_context,
            key=moves_embed_pre_pool,
            value=moves_embed_pre_pool,
            preprocessed_value=False,
            indexed=True,
        )

        # party pokemon embedding
        moves = self.moves_embedding(tf.cast(obs["party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["party_attributes"]
        party_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        attended_party = self.party_attention(
            query=battle_context,
            key=party_info,
            value=party_info,
            indexed=True,
            preprocessed_value=False
        )

        items = self.items_embedding(tf.cast(obs['item_ids'], tf.int64))
        item_quantities = tf.expand_dims(obs['item_quantities'], axis=-1)
        items_info = tf.concat([items, item_quantities], axis=-1)
        items_embed = self.items_mlp(items_info)
        items_embed = tf.reduce_max(items_embed, axis=-2)

        additional_ram_info = obs["ram"]

        concat = tf.concat([
            screen_embed,
            items_embed,
            attended_moves,
            battle_context,
            attended_party,
            sent_out_index,
            additional_ram_info,
        ], axis=-1)

        return self.final_mlp(concat)

    def batch_input(  # sent_out_party_index ???????????????
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
        for conv in self.screen_conv_layers[1:]:
            conv_out = conv(conv_out)
            conv_out = self.conv_activation(conv_out)
        conv_out_flat = snt.Flatten(1)(conv_out)
        conv_out = tf.reshape(conv_out_flat, tf.concat([shape[:2], [-1]], axis=0))
        screen_embed = self.screen_embedding(conv_out)

        in_battle_mask = tf.cast(obs["is_in_battle"], dtype=tf.float32)

        # sent out party embed
        moves = self.moves_embedding(tf.cast(obs["sent_out_party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["sent_out_party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_info = tf.concat([moves, pps, pps_mask], axis=-1)
        moves_embed_pre_pool = self.move_mlp(moves_info)
        moves_embed = tf.reduce_max(moves_embed_pre_pool, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["sent_out_party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["sent_out_party_attributes"]

        sent_out_index = tf.one_hot(tf.cast(obs["sent_out_party_index"], tf.int64), 6, dtype=tf.float32)[:, :, 0]
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

        battle_context = tf.concat([sent_out_party_info, sent_out_opp_info], axis=-1)

        attended_moves = self.move_attention(
            query=battle_context,
            key=moves_embed_pre_pool,
            value=moves_embed_pre_pool,
            preprocessed_value=False,
            indexed=True,
        )

        # party pokemon embedding
        moves = self.moves_embedding(tf.cast(obs["party_move_ids"], tf.int64))
        pps = tf.expand_dims(obs["party_pps"], axis=-1)
        pps_mask = tf.cast(pps > 0, dtype=tf.float32)
        moves_embed = self.move_mlp(tf.concat([moves, pps, pps_mask], axis=-1))
        moves_embed = tf.reduce_max(moves_embed, axis=-2)

        types = tf.reduce_sum(self.types_embedding(tf.cast(obs["party_type_ids"], tf.int64)), axis=-2)  # (e,)
        attributes = obs["party_attributes"]
        party_info = self.pokemon_mlp(tf.concat([moves_embed, types, attributes], axis=-1))

        attended_party = self.party_attention(
            query=battle_context,
            key=party_info,
            value=party_info,
            indexed=True,
            preprocessed_value=False
        )

        items = self.items_embedding(tf.cast(obs['item_ids'], tf.int64))
        item_quantities = tf.expand_dims(obs['item_quantities'], axis=-1)
        items_info = tf.concat([items, item_quantities], axis=-1)
        items_embed = self.items_mlp(items_info)
        items_embed = tf.reduce_max(items_embed, axis=-2)

        additional_ram_info = obs["ram"]

        concat = tf.concat([
            screen_embed,
            items_embed,
            attended_moves,
            battle_context,
            attended_party,
            sent_out_index,
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
        return tf.math.square(vf_targets - self._values)
        # return self.value_head.loss(vf_targets)

    def get_initial_state(self):
        return (np.zeros(2, dtype=np.float32),)

    def get_metrics(self):
        return {}
