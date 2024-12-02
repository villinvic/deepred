import abc
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable, Any, Tuple, Union, List, Dict

import cv2
import numpy as np
import tree
from PIL import Image
from gymnasium import spaces
from gymnasium.spaces import Box
from deepred.polaris_env.enums import Map, EventFlag, BagItem, ProgressionEvents
from deepred.polaris_env.gamestate import GameState


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

    def initialise(
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
            size: int = 1,
            scale: float = 1,
            domain: Tuple = (-1e8, 1e8),
    ):
        """
        :param extractor: Function to call on a gamestate to grab the observation of interest.
        :param nature: Any of ObsType.
        :param size: size of the observation. Number of observed values of binary and continuous observations,
            number of classes for categorical observations.
        :param scale: scale for continuous observations. The observed value will be (x * scale).
        :param domain: domain for continuous observations. The observation will be clipped into this domain before being
            scaled.
        """

        super().__init__(extractor)
        self.nature = nature
        self.size = size
        self.scale = scale
        low, high = domain
        self.domain = (low * scale, high * scale)

    def gym_spec(self) -> spaces.Box:

        if self.nature in (ObsType.BINARY, ObsType.CATEGORICAL):
            # For now, we onehot categoricals from here, but we could one hot on the model side with tensorflow.
            return spaces.Box(0, 1, (self.size,))
        elif self.nature == ObsType.CONTINUOUS:
            return spaces.Box(*self.domain, (self.size,))
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
                input_array[self._offset : self._offset + self.size] = 0
                input_array[self._offset + observed] = 1
        elif self.nature == ObsType.CONTINUOUS:
                input_array[self._offset: self._offset + self.size] = np.clip(observed, *self.domain) * self.scale
        else:
            raise NotImplementedError(f"Observation type {self.nature} is not supported.")

class PixelsObservation(Observation):

    def __init__(
            self,
            extractor: Callable[[GameState], Any],
            framestack: int,
            stack_oldest_only: int,
            downscaled_shape: Tuple
    ):

        super().__init__(extractor)
        self.framestack = framestack
        self.stack_oldest_only = stack_oldest_only
        h, w = downscaled_shape
        self.downscaled_shape = (w, h) # we reverse w and h for opencv.
        if self.stack_oldest_only:
            h *= 2
        else:
            h *= framestack
        self.stacked_shape = (h, w)

        # needed if we use stack_oldest_only
        if self.stack_oldest_only:
            self._pixel_history = np.zeros(
                (h*framestack, w), dtype=np.uint8
            )

    def gym_spec(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=self.stacked_shape, dtype=np.uint8)

    def to_obs(
            self,
            gamestate: GameState,
            input_array: np.ndarray
    ):
        pixels = self._extractor(gamestate)
        w, h = self.downscaled_shape
        if pixels.shape == self.downscaled_shape:
            downscaled = pixels
        else:
            downscaled = cv2.resize(
                pixels,
                self.downscaled_shape,
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
            input_array[h:] = input_array[:-h]
            input_array[:h] = downscaled


class PolarisRedObservationSpace:

    def __init__(
            self,
            downscaled_screen_shape: Tuple,
            framestack: int,
            stack_oldest_only: bool,
            observed_ram: Tuple[str],
            observed_items: Tuple[BagItem],
    ):
        """
        Manages the observation interface between the game and agent.
        :param downscaled_screen_shape: final shape of the pixel observation after downscaled.
        :param framestack: number of frames to stack on top of each other for pixel observations.
        :param stack_oldest_only: whether to only stack the oldest frame in the stack on top of the current frame.
        :param observed_ram: list of ram observations to include into the final observation.
        :param observed_items: list of bag items to include into the final observation: c.f. the BagItem enum.
        """

        def extract_bag_items(gamestate: GameState):
            bag_items = gamestate.bag_items
            return [bag_items.get(key, 0) for key in observed_items]

        ram_observations = dict(
            badges=RamObservation(
                extractor=lambda gamestate: gamestate.badges,
                nature=ObsType.BINARY,
                size=8
            ),
            money=RamObservation(
                extractor=lambda gamestate: gamestate.player_money,
                nature=ObsType.CONTINUOUS,
                scale=1e-4,
                domain=(0., 20_000.)
            ),
            species_seen=RamObservation(
                extractor=lambda gamestate: gamestate.species_seen_count,
                nature=ObsType.CONTINUOUS,
                scale=1e-2,
                domain=(0., 151.)
            ),
            species_caught=RamObservation(
                extractor=lambda gamestate: gamestate.species_caught_count,
                nature=ObsType.CONTINUOUS,
                scale=4e-2,
                domain=(0., 151.)
            ),
            party_hp=RamObservation(
                extractor=lambda gamestate: gamestate.party_hp,
                nature=ObsType.CONTINUOUS,
                size=6,
                domain=(0., 1.)
            ),
            party_level=RamObservation(
                extractor=lambda gamestate: gamestate.party_level,
                nature=ObsType.CONTINUOUS,
                size=6,
                scale=1/50,
                domain=(0., 100.)
            ),
            in_battle=RamObservation(
                extractor=lambda gamestate: gamestate.is_in_battle,
                nature=ObsType.BINARY,
            ),
            sent_out=RamObservation(
                extractor=lambda gamestate: gamestate.sent_out,
                nature=ObsType.CATEGORICAL,
                size=6,
            ),
            bag_items=RamObservation(
                extractor=extract_bag_items,
                nature=ObsType.CONTINUOUS,
                size=len(observed_items),
                scale=1/4,
                domain=(0., 8.)
            ),
            event_flags=RamObservation(
                extractor=lambda gamestate: gamestate.event_flags,
                nature=ObsType.BINARY,
                size=len(ProgressionEvents),
            ),
            position=RamObservation(
                extractor=lambda gamestate: [gamestate.pos_x, gamestate.pos_y],
                nature=ObsType.CONTINUOUS,
                size=2,
            ),
            map_id=RamObservation(
                extractor=lambda gamestate: gamestate.map,
                nature=ObsType.CATEGORICAL,
                size=len(Map)
            )
        )

        offset = 0
        for name, observation in list(ram_observations.items()):
            if name not in observed_ram:
                del ram_observations[name]
            else:
                observation.initialise(offset)
                offset += observation.gym_spec().shape[0]

        pixel_observation = PixelsObservation(
                extractor=lambda gamestate: gamestate.screen,
                framestack=framestack,
                stack_oldest_only=stack_oldest_only,
                downscaled_shape=downscaled_screen_shape
        )
        pixel_observation.initialise()

        minimap_observation = PixelsObservation(
            extractor=lambda gamestate: gamestate.minimap
        )

        self.observations = dict(
            ram=ram_observations,
            main_screen=pixel_observation,
            minimap=minimap_observation,
            minimap_sprite=minimap_sprite_observation,
            minimap_warp=minimap_warp_observation,
            map_ids=last_10_map_ids_observation,
            map_step_since=last_10_map_step_since_observation,
            item_ids=all_item_ids_observation,
            item_quantity=items_quantity_observation,
            pokemon_ids=all_pokemon_ids_observation,
            pokemon_type_ids=all_pokemon_types_observation,
            pokemon_move_ids=all_move_ids_observation,
            pokemon_move_pps= all_move_pps_observation,
            pokemon_all=all_pokemon_observaton,
            event_ids=all_event_ids_observation,
            event_step_since=all_event_step_since_observation,
        )

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
            flatten = False,
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
                            return Box(low, high, shape, gym_specs[0].dtype)

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
            observations: Dict
    ):
        """
        Dumps the observation to the designated path. Pixel observations are output as images.
        :param dump_path:
        :param observations:
        :return:
        """

        for observation_name, extractor in self.observations.items():
            output_file = dump_path.with_name(dump_path.stem + f"_{observation_name}")
            if isinstance(extractor, PixelsObservation):
                image_file = output_file.with_suffix(".png")
                Image.fromarray(observations[observation_name]).save(image_file)
            elif isinstance(extractor, dict):
                self.dump_observations(
                    dump_path.with_name(dump_path.stem + f"_{observation_name}"), extractor
                )
            else:
                log_file = output_file.with_suffix(".log")
                with open(log_file, "w") as f:
                    f.write(str(observations[observation_name]))





