import abc
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable, Any, Tuple, Union, List, Dict

import cv2
import gymnasium.spaces
import numpy as np
import tree
from PIL import Image
from gymnasium import spaces
from gymnasium.spaces import Box

from deepred.polaris_env.pokemon_red.enums import Map, BagItem, Pokemon, PokemonType, Move, ProgressionFlag
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.map_dimensions import MapDimensions


class ObsType(Enum):
    BINARY = 0
    CATEGORICAL = 1
    CONTINUOUS = 2

class Observation(abc.ABC):

    def __init__(
            self,
            extractor: Callable[[GameState], Any],
    ):
        self._extractor = extractor
        self._offset = 0

    @abstractmethod
    def gym_spec(self) -> spaces.Box:
        """
        :return: the gymnasium spec of this observation.
        """
        pass

    def set_offset(
            self,
            offset: int = 0,
    ):
        """
        Initialises the ram observation with the provided offset.
        The main ram observation array will be populated with this observation at the given offset.
        :param offset: offset of the observation in the ram observation array.
        """
        self._offset = offset

    @abstractmethod
    def to_obs(
            self,
            gamestate: GameState,
            input_array: np.ndarray
    ):
        """
        :param gamestate: state from which we will extract information
        :param input_array: The input array to write to.
        """
        pass


class RamObservation(Observation):

    def __init__(
            self,
            extractor: Callable[[GameState], Any],
            nature: ObsType,
            size: int | Tuple[int, int] = 1,
            scale: float = 1,
            domain: Tuple = (-1e8, 1e8),
            preprocess=True,
    ):
        """
        :param extractor: Function to call on a gamestate to grab the observation of interest.
        :param nature: Any of ObsType.
        :param size: size of the observation. Number of observed values of binary and continuous observations,
            number of classes for categorical observations.
        :param scale: scale for continuous observations. The observed value will be (x * scale).
        :param domain: domain for continuous observations. The observation will be clipped into this domain before being
            scaled.
        :param preprocess: we might need to not preprocess categorical inputs, if False, leaves the categorical value as
        is and stores the highest value possible in the box space's 'high' attribute. The size is used if multiple
        similar values should be observed.
        # TODO: confusing for categorical values: dim should be one, change the domain instead
        """

        super().__init__(extractor)
        self.nature = nature
        self.size = size
        self.shape = (size,) if isinstance(size, int) else size
        self.scale = scale
        low, high = domain
        self.domain = (low * scale, high * scale)
        self.preprocess = preprocess

    def gym_spec(self) -> spaces.Box:
        if self.nature == ObsType.CATEGORICAL:
            if self.preprocess:
                shape = (int(self.domain[1]),)
                if self.size != 1:
                    shape =  self.shape + shape
                return spaces.Box(0, 1, shape, dtype=np.uint8)
            else:
                # automatically infer best datatype:
                high = int(self.domain[1])
                if high < 256:
                    dtype = np.uint8
                else:
                    dtype = np.uint16
                return spaces.Box(0, high, self.shape, dtype=dtype)

        elif self.nature == ObsType.BINARY:
            # For now, we onehot categoricals from here, but we could one hot on the model side with tensorflow.
            return spaces.Box(0, 1, self.shape, dtype=np.uint8)
        elif self.nature == ObsType.CONTINUOUS:
            return spaces.Box(*self.domain, self.shape, dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation type {self.nature} is not supported.")

    def to_obs(
            self,
            gamestate: GameState,
            input_array: np.ndarray
    ):
        observed = self._extractor(gamestate)
        if self.nature == ObsType.BINARY:
            input_array[self._offset: self._offset + self.size] = observed
        elif self.nature == ObsType.CATEGORICAL:
            if self.preprocess:
                input_array[self._offset: self._offset + self.size] = 0
                input_array[self._offset + observed] = 1
            else:
                assert self._offset == 0
                input_array[:] = observed
        elif self.nature == ObsType.CONTINUOUS:
            if isinstance(self.size, tuple):
                assert self._offset == 0
                input_array[:] = observed
            else:
                input_array[self._offset: self._offset + self.size] = np.clip(observed, *self.domain) * self.scale
        else:
            raise NotImplementedError(f"Observation type {self.nature} is not supported.")

class PixelsObservation(Observation):

    def __init__(
            self,
            extractor: Callable[[GameState], Any],
            downscaled_shape: Tuple,
            framestack: int = 1,
            stack_oldest_only: bool = False,
            dtype=np.uint8,
    ):

        super().__init__(extractor)
        self.framestack = framestack
        self.stack_oldest_only = stack_oldest_only
        self.downscaled_shape = downscaled_shape
        if len(downscaled_shape) > 2:
            self.stacked_shape = downscaled_shape
        else:

            h, w = downscaled_shape
            self.opencv_downscaled_shape = (w, h) # we reverse w and h for opencv.
            if self.stack_oldest_only:
                h *= 2
            else:
                h *= framestack
            self.stacked_shape = (h, w)

            # needed if we use stack_oldest_only
            if self.stack_oldest_only:
                self._pixel_history = np.zeros(
                    (h*framestack, w), dtype=dtype
                )
        self.dtype = dtype

    def gym_spec(self) -> spaces.Box:
        high = dtype_sizes[self.dtype]**2 - 1
        return spaces.Box(low=0, high=high, shape=self.stacked_shape, dtype=self.dtype)

    def to_obs(
            self,
            gamestate: GameState,
            input_array: np.ndarray
    ):
        pixels = self._extractor(gamestate)
        h, w = self.downscaled_shape[-2:]
        if pixels.shape == self.stacked_shape:
            downscaled = pixels
        else:
            downscaled = cv2.resize(
                pixels,
                self.opencv_downscaled_shape,
                interpolation=cv2.INTER_AREA,
            )

        if self.stack_oldest_only:
            # Here, we only stack the most recent pixels to the oldest one in the stack.
            # This may be useful to limit input dimensions.

            self._pixel_history[h:] = self._pixel_history[:-h]
            self._pixel_history[:h] = downscaled

            input_array[:h] = downscaled
            input_array[h:] = self._pixel_history[-h:]

        else:
            if self.framestack > 1:
                input_array[h:] = input_array[:-h]
            input_array[:h] = downscaled


class PolarisRedObservationSpace:

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
        """

        # TODO: allow auto inference of dimensions, rather than passing len of ...
        base_ram_observations = dict(
            # badges=RamObservation(
            #     extractor=lambda gamestate: gamestate.badges,
            #     nature=ObsType.BINARY,
            #     size=8
            # ),
            # money=RamObservation(
            #     extractor=lambda gamestate: gamestate.player_money,
            #     nature=ObsType.CONTINUOUS,
            #     scale=2e-4,
            #     domain=(0., 30_000.)
            # ),
            # current_checkpoint=RamObservation(
            #     extractor=lambda gamestate: gamestate.current_checkpoint,
            #     nature=ObsType.CATEGORICAL,
            #     domain=(0,len(dummy_gamestate._additional_memory.pokecenter_checkpoints.pokecenter_ids)),
            # ),
            # visited_pokecenters=RamObservation(
            #     extractor=lambda gamestate: gamestate.visited_pokemon_centers,
            #     nature=ObsType.BINARY,
            #     size=len(dummy_gamestate._additional_memory.pokecenter_checkpoints.pokecenter_ids),
            # ),
            # field_moves=RamObservation(
            #     extractor=lambda gamestate: gamestate.field_moves,
            #     nature=ObsType.BINARY,
            #     size=len(dummy_gamestate.field_moves),
            # ),
            # have_hms=RamObservation(
            #     extractor=lambda gamestate: gamestate.hms,
            #     nature=ObsType.BINARY,
            #     size=len(dummy_gamestate.hms),
            # ),
            # battle_type=RamObservation(
            #     extractor=lambda gamestate: gamestate.battle_type,
            #     nature=ObsType.CATEGORICAL,
            #     domain=(0, 3),
            # ),
            # party_count=RamObservation(
            #     extractor=lambda gamestate: gamestate.party_count,
            #     nature=ObsType.CONTINUOUS,
            #     scale=1/6,
            #     domain=(1/6, 1.)
            # ),
            # party_full=RamObservation(
            #     extractor=lambda gamestate: int(gamestate.party_count == 6),
            #     nature=ObsType.BINARY,
            # ),
            # better_pokemon_in_box=RamObservation(
            #     extractor=lambda gamestate: gamestate.has_better_pokemon_in_box,
            #     nature=ObsType.BINARY,
            # ),
            # bag_full=RamObservation(
            #     extractor=lambda gamestate: gamestate.bag_count == 20,
            #     nature=ObsType.BINARY
            # ),
            # bag_count=RamObservation(
            #     extractor=lambda gamestate: gamestate.bag_count,
            #     nature=ObsType.CONTINUOUS,
            #     scale=1/20,
            #     domain=(0., 20.),
            # ),
            coordinates=RamObservation(
                extractor=lambda gamestate: gamestate.scaled_coordinates,
                nature=ObsType.CONTINUOUS,
                size=2,
                scale=1,
                domain=(0., 1.),
            ),
            # Optional addons
            # battle turns
        )

        offset = 0
        for name, observation in list(base_ram_observations.items()):
            observation.set_offset(offset)
            offset += observation.gym_spec().shape[0]

        pixel_observation = PixelsObservation(
                extractor=lambda gamestate: gamestate.screen,
                downscaled_shape=downscaled_screen_shape,
                framestack=framestack,
                stack_oldest_only=stack_oldest_only,
        )

        feature_maps_observation = PixelsObservation(
            extractor=lambda gamestate: gamestate.feature_maps,
            downscaled_shape=dummy_gamestate.feature_maps.shape
        )

        sprite_map_observation = PixelsObservation(
            extractor=lambda gamestate: gamestate.sprite_map,
            downscaled_shape=dummy_gamestate.sprite_map.shape,
            dtype=np.uint16,
        )

        warp_map_observation = PixelsObservation(
            extractor=lambda gamestate: gamestate.warp_map,
            downscaled_shape=dummy_gamestate.warp_map.shape,
            dtype=np.uint16,
        )

        last_visited_map_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.last_visited_maps,
            nature=ObsType.CATEGORICAL,
            size=dummy_gamestate._additional_memory.map_history.map_history_length,
            domain=(0, len(Map)),
            preprocess=False # we compute embeddings with the model
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

        # I think this is not needed, as the id provides no additional information when we are already observing stats, moves, etc.
        pokemon_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.pokemons,
            nature=ObsType.CATEGORICAL,
            size=12,
            domain=(0, len(Pokemon)),
            preprocess=False
        )

        pokemon_type_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.pokemon_types,
            nature=ObsType.CATEGORICAL,
            size=(12, 2),
            domain=(0, len(PokemonType)),
            preprocess=False
        )

        pokemon_move_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.pokemon_moves,
            nature=ObsType.CATEGORICAL,
            size=(12, 4),
            domain=(0, len(Move)),
            preprocess=False
        )

        pokemon_pps_observation = RamObservation(
            extractor=lambda gamestate: gamestate.pokemon_pps,
            nature=ObsType.CONTINUOUS,
            size=(12, 4),
            scale=1/30,
            domain=(0., 30.),
        )

        pokemon_stats_observation = RamObservation(
            extractor=lambda gamestate: gamestate.pokemon_attributes,
            nature=ObsType.CONTINUOUS,
            size=dummy_gamestate.pokemon_attributes.shape,
            scale=1., # everything was prescaled
            domain=(0., 1.),
        )

        recent_event_ids_observation = RamObservation(
            extractor=lambda gamestate: gamestate.last_triggered_flags,
            nature=ObsType.CATEGORICAL,
            size=dummy_gamestate._additional_memory.flag_history.flag_history_length,
            domain=(0, len(ProgressionFlag)),
            preprocess=False
        )

        recent_event_ids_age_observation = RamObservation(
            extractor=lambda gamestate: gamestate.last_triggered_flags_age,
            nature=ObsType.CONTINUOUS,
            size=dummy_gamestate._additional_memory.flag_history.flag_history_length,
            scale=1e-4,
            domain=(0., 1.),
        )

        # recent pokemon centers, last checkpoint

        self.observations = dict(
            ram=base_ram_observations,
            main_screen=pixel_observation,
            feature_maps=feature_maps_observation,
            sprite_map=sprite_map_observation,
            warp_map=warp_map_observation,
            map_ids=last_visited_map_ids_observation,
            # map steps since ?
            item_ids=bag_item_ids_observation,
            item_quantities=bag_item_counts_observation,
            #pokemon_ids=pokemon_ids_observation,
            pokemon_type_ids=pokemon_type_ids_observation,
            pokemon_move_ids=pokemon_move_ids_observation,
            pokemon_move_pps= pokemon_pps_observation,
            pokemon_stats=pokemon_stats_observation,
            recent_event_ids=recent_event_ids_observation,
            recent_event_ids_age=recent_event_ids_age_observation,
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

