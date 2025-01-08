from pathlib import Path
from typing import Tuple, Dict, Callable, List, Union

import gymnasium
import numpy as np
import tree
from PIL.Image import Image
from gymnasium import spaces
from gymnasium.spaces import Box

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.observation_space import RamObservation, ObsType, PixelsObservation, Observation
from deepred.polaris_env.pokemon_red.enums import BattleType, FixedPokemonType, Move, BagItem


class PolarisRedArenaObservationSpace:

    def __init__(
            self,
            downscaled_screen_shape: Tuple,
            framestack: int,
            stack_oldest_only: bool,
            dummy_gamestate: GameState,
    ):
        """
        Manages the observation interface between the game and agent.
        :param downscaled_screen_shape: final shape of the pixel observation after downscaled.
        :param framestack: number of frames to stack on top of each other for pixel observations.
        :param stack_oldest_only: whether to only stack the oldest frame in the stack on top of the current frame.

        This is a simpler version than PolarisRedObservationSpace, where all observations irrelevant to battles were
        evinced.
        """

        base_ram_observations = dict(
            party_count=RamObservation(
                extractor=lambda gamestate: gamestate.party_count - 1,
                nature=ObsType.CATEGORICAL,
                domain=(0, 5)
            ),
            bag_count=RamObservation(
                extractor=lambda gamestate: gamestate.bag_count,
                nature=ObsType.CONTINUOUS,
                scale=1/20,
                domain=(0., 20.),
            ),
            hp_frac=RamObservation(
                extractor=lambda gamestate: gamestate.party_count - np.sum(gamestate.party_hp[:gamestate.party_count]),
                nature = ObsType.CONTINUOUS,
                scale = 1/2,
                domain = (0., 6.),
            ),
            pp_frac=RamObservation(
                extractor=lambda gamestate: np.mean(gamestate.party_pps[:gamestate.party_count]),
                nature = ObsType.CONTINUOUS,
                scale = 1/20,
                domain = (0., 20.),
            ),
            battle_type=RamObservation(
                extractor=lambda gamestate: gamestate.battle_type,
                nature=ObsType.CATEGORICAL,
                domain=(0, len(BattleType)),
            ),
        )


        offset = 0
        for name, observation in list(base_ram_observations.items()):
            observation.set_offset(offset)
            offset += observation.gym_spec().shape[0]

        # No preprocessing obs

        pixel_observation = PixelsObservation(
                extractor=lambda gamestate: gamestate.screen,
                downscaled_shape=downscaled_screen_shape,
                framestack=framestack,
                stack_oldest_only=stack_oldest_only,
        )


        bag_item_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.ordered_bag_items,
            nature=ObsType.CATEGORICAL,
            size=20,
            domain=(0, len(BagItem)),
            preprocess=False # we compute embeddings with the model
        )

        bag_item_counts_observation = RamObservation(
            extractor=lambda gamestate: gamestate.ordered_bag_items,
            nature=ObsType.CONTINUOUS,
            size=20,
            scale=1/10,
            domain=(0., 10.),
        )

        party_type_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.party_types,
            nature=ObsType.CATEGORICAL,
            size=(6, 2),
            domain=(0, len(FixedPokemonType)),
            preprocess=False
        )

        sent_out_party_type_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_party_types,
            nature=ObsType.CATEGORICAL,
            size=2,
            domain=(0, len(FixedPokemonType)),
            preprocess=False
        )

        sent_out_opp_type_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_opponent_types,
            nature=ObsType.CATEGORICAL,
            size=2,
            domain=(0, len(FixedPokemonType)),
            preprocess=False
        )

        party_move_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.party_moves,
            nature=ObsType.CATEGORICAL,
            size=(6, 4),
            domain=(0, len(Move)),
            preprocess=False
        )

        sent_out_party_move_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_party_moves,
            nature=ObsType.CATEGORICAL,
            size=4,
            domain=(0, len(Move)),
            preprocess=False
        )

        sent_out_opp_move_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_opponent_moves,
            nature=ObsType.CATEGORICAL,
            size=4,
            domain=(0, len(Move)),
            preprocess=False
        )

        party_pps_observation = RamObservation(
            extractor=lambda gamestate: gamestate.party_pps,
            nature=ObsType.CONTINUOUS,
            size=(6, 4),
            scale=1/20,
            domain=(0., 20.),
        )

        sent_out_party_pps_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_party_pps,
            nature=ObsType.CONTINUOUS,
            size=4,
            scale=1/20,
            domain=(0., 20.),
        )

        sent_out_opp_pps_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_opponent_pps,
            nature=ObsType.CONTINUOUS,
            size=4,
            scale=1/20,
            domain=(0., 20.),
        )

        party_attributes_observation = RamObservation(
            extractor=lambda gamestate: gamestate.party_attributes,
            nature=ObsType.CONTINUOUS,
            size=dummy_gamestate.party_attributes.shape,
            scale=1., # everything was prescaled
            domain=(0., 3.),
        )

        sent_out_party_attributes_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_party_attributes,
            nature=ObsType.CONTINUOUS,
            size=dummy_gamestate.sent_out_party_attributes.shape,
            scale=1., # everything was prescaled
            domain=(0., 3.),
        )

        sent_out_opp_attributes_observation = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out_opponent_attributes,
            nature=ObsType.CONTINUOUS,
            size=dummy_gamestate.sent_out_opponent_attributes.shape,
            scale=1., # everything was prescaled
            domain=(0., 3.),
        )


        sent_out_party_index = RamObservation(
            extractor=lambda gamestate: gamestate.sent_out,
            nature=ObsType.CATEGORICAL,
            size=1,
            domain=(0, 5),
            preprocess=False
        )
        # recent pokemon centers, last checkpoint

        self.observations = dict(
            ram=base_ram_observations,
            main_screen=pixel_observation,
            item_ids=bag_item_ids_observation,
            item_quantities=bag_item_counts_observation,

            party_type_ids=party_type_ids_observation,
            sent_out_party_type_ids=sent_out_party_type_ids_observation,
            sent_out_opp_type_ids=sent_out_opp_type_ids_observation,

            party_pps=party_pps_observation,
            sent_out_party_pps=sent_out_party_pps_observation,
            sent_out_opp_pps=sent_out_opp_pps_observation,

            party_move_ids=party_move_ids_observation,
            sent_out_party_move_ids=sent_out_party_move_ids_observation,
            sent_out_opp_move_ids=sent_out_opp_move_ids_observation,

            party_attributes=party_attributes_observation,
            sent_out_party_attributes=sent_out_party_attributes_observation,
            sent_out_opp_attributes=sent_out_opp_attributes_observation,

            sent_out_party_index=sent_out_party_index

        )

        # TODO: adapt and test for dtypes, and various types of observations
        self.inject = self._build_observation_op(self.observations)

        observation_gym_specs = tree.map_structure(
            lambda obs: obs.gym_spec(),
            self.observations
        )

        self.unflattened_gym_spec = self._to_gym_space(observation_gym_specs)
        self.sample_unflattened_obs = self.unflattened_gym_spec.sample()

        # flattened gym spec
        self.gym_spec = self._to_gym_space(observation_gym_specs, flatten=True)
        self.sample_obs = self.gym_spec.sample()

    def unflatten(
            self,
            flat_observation: np.ndarray
    ):
        """
        :param flat_observation: A flat observation yielded by this observation space.
        :return: The unflattened observation
        """
        # TODO: this does not work, the leaves have to be converted into lists, but this is costly.
        unflattened = {
            "pixels": flat_observation["pixels"],
            "ram": tree.unflatten_as(self.sample_unflattened_obs["ram"], list(flat_observation["ram"]))
        }

        return unflattened

    def _build_observation_op(
            self,
            observations: Dict[str, Union[Dict, Observation]]
    ) -> Callable[[GameState, Dict[str, np.ndarray]], None]:
        """
        :param observations: dictionary of observations we want to feed to the agent.
        :return: the observation building function.
        """
        def op(
                gamestate: GameState,
                input_dict: Dict[str, np.ndarray]
        ):

            for k, input_array in input_dict.items():
                tree.map_structure(
                    lambda obs: obs.to_obs(
                        gamestate,
                        input_array
                    ),
                    observations[k]
                )

        return op

    def _to_gym_space(
            self,
            observation_spaces,
            flatten=False,
    ) -> spaces.Dict:
        """
        Recursively converts a dictionary of dictionaries into a gym.spaces.Dict.

        """
        gym_space_dict = {}
        for key, value in observation_spaces.items():
            if isinstance(value, dict):
                if flatten and len(value) > 1:
                    # check if all nodes have same observation type, then we flatten.
                    values = list(value.values())
                    if all(
                            [isinstance(v, spaces.Box) for v in values[1:]]
                    ):
                        def concat_gym_boxes(gym_specs: List[spaces.Box]):
                            low = np.concatenate([gym_spec.low for gym_spec in gym_specs], axis=0)
                            high = np.concatenate([gym_spec.high for gym_spec in gym_specs], axis=0)
                            shape = low.shape
                            dtype = gym_specs[0].dtype
                            for gym_spec in gym_specs[1:]:
                                if dtype_sizes[gym_spec.dtype] > dtype_sizes[dtype]:
                                    dtype = gym_spec.dtype

                            return Box(low, high, shape, dtype)

                        gym_space_dict[key] = concat_gym_boxes(values)

                    else:
                        # attempt again without flattening
                        gym_space_dict[key] = self._to_gym_space(value, flatten=False)

                else:
                    # Recursively convert nested dictionaries
                    gym_space_dict[key] = self._to_gym_space(value)
            else:
                gym_space_dict[key] = value
        return spaces.Dict(gym_space_dict)

    def dump_observations(
            self,
            dump_path: Path,
            observations: Dict,
            observation_space: gymnasium.spaces.Space | None = None,
    ):
        """
        Dumps the observation to the designated path. Pixel observations are output as images.
        :param dump_path: path to dump the observations.
        :param observations: observations to dump .
        :param observation_space: observation space based on which we dump.
        """
        if observation_space is None:
            observation_space = self.observations


        # TODO: infinite recursion -> we interate over self.observations !!!
        for observation_name, extractor in observation_space.items():
            output_file = dump_path.with_name(dump_path.stem + f"_{observation_name}")
            if isinstance(extractor, PixelsObservation):
                pixels = observations[observation_name]
                if len(pixels.shape) == 3:
                    for i, subpixels in enumerate(pixels):
                        image_file = output_file.with_name(output_file.stem + f"_{i}").with_suffix(".png")
                        Image.fromarray(subpixels).save(image_file)
                else:
                    print(observation_name, pixels.shape, np.max(pixels))
                    image_file = output_file.with_suffix(".png")
                    Image.fromarray(pixels).save(image_file)
            elif isinstance(extractor, dict):
                self.dump_observations(
                    dump_path.with_name(dump_path.stem + f"_{observation_name}"), observations[observation_name], extractor
                )
            else:
                log_file = output_file.with_suffix(".log")
                with open(log_file, "w") as f:
                    f.write(str(observations))


dtype_sizes = {
    np.float32: 128,
    np.dtype(np.float32): 128,
    np.int32: 64,
    np.dtype(np.int32): 64,
    np.uint16: 16,
    np.dtype(np.uint16): 16,
    np.uint8 : 8,
    np.dtype(np.uint8): 8,
}

