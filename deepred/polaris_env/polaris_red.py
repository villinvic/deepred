import sys
import time
import uuid
import os
from functools import partial
from typing import List, Callable, Any, Union, SupportsFloat, Dict, Optional, Tuple

import cv2
from collections import defaultdict
from math import floor, sqrt
import json
from pathlib import Path

import plotly
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import ObsType, ActType
from polaris.environments import PolarisEnv
from pyboy import PyBoy
from pyboy.utils import WindowEvent

import mediapy

from gymnasium import spaces

from deepred.polaris_env.action_space import PolarisRedActionSpace
from deepred.polaris_env.enums import *
from deepred.polaris_env.gb_console import GBConsole
from deepred.polaris_env.metrics import PolarisRedMetrics
from deepred.polaris_env.observation_space import PolarisRedObservationSpace
from deepred.polaris_env.rewards import PolarisRedRewardFunction, Goals


class PolarisRed(PolarisEnv):
    env_id = "PolarisRed"

    # TODO: https://github.com/pwhiddy/pokerl-map-viz/
    # visualise all instances in one single red map.

    def __init__(
            self,
            env_index=-1,
            game_path: str= "faster_red.gbc",
            episode_length=2048,
            enable_start: bool = True,
            enable_pass: bool = True,
            downscaled_screen_shape: Tuple = (72, 80),
            framestack: int = 3,
            stack_oldest_only: bool = False,
            observed_ram: Tuple[str] = ("badges",),
            observed_items: Tuple[BagItem] = (BagItem.POKE_BALL, BagItem.POTION),
            reward_scales: dict  | None = None,
            savestate: Union[None, str] = None,
            session_path: str = "red_tests",
            render: bool = True,
            record: bool = False,
            speed_limit: int = 1,
            record_skipped_frame: bool = False,
            ** config
    ):
        super().__init__(env_index, **config)
        self._agent_ids = {0}
        self.empty_info_dict = {0: {}}

        self.render = env_index == 0 and render
        self.session_path = Path(session_path)
        self.step_count = 0
        self.episode_length = episode_length

        self.session_path.mkdir(exist_ok=True)

        self.input_interface = PolarisRedActionSpace(
            enable_start=enable_start,
            enable_pass=enable_pass
        )
        self.observation_interface = PolarisRedObservationSpace(
            downscaled_screen_shape=downscaled_screen_shape,
            framestack=framestack,
            stack_oldest_only=stack_oldest_only,
            observed_ram=observed_ram,
            observed_items=observed_items
        )

        self.action_space = self.input_interface.gym_spec
        self.observation_space = self.observation_interface.gym_spec

        self._console_maker = partial(
            GBConsole,
            console_id=self.env_index,
            game_path=game_path,
            render=self.render,
            speed_limit=speed_limit,
            record=record,
            record_skipped_frames=record_skipped_frame,
            output_dir=self.session_path / Path(f"console_{self.env_index}"),
            savestate=savestate,
            **config
        )
        self.console: Union[GBConsole, None] = None

        self.metrics : Union[PolarisRedMetrics, None] = None
        self.reward_scales = reward_scales
        self.reward_function: Union[PolarisRedRewardFunction, None] = None
        self.input_dict: Union[Dict, None] = None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: dict[str, dict] = None,
    ) -> Tuple[Dict[int, dict], dict]:
        if self.console is None:
            self.console = self._console_maker()

        gamestate = self.console.reset()
        self.metrics = PolarisRedMetrics()
        self.reward_function = PolarisRedRewardFunction(
            reward_scales=self.reward_scales,
            count_based_exploration_scales=options[0]["count_based_exploration_scales"],
            inital_gamestate=gamestate,
        )
        self.input_dict = self.observation_space.sample()
        self.observation_interface.inject(
            gamestate,
            self.input_dict
        )

        self.step_count = 0

        # No need to make a copy of the input dict, we perform a deepcopy in polaris.
        return {0: self.input_dict}, self.empty_info_dict


    def step(
        self, action_dict: Dict[int, int]
    ) -> Tuple[dict, dict, dict, dict, dict]:

        event = self.input_interface.get_event(action_dict[0])
        gamestate = self.console.step_event(event)

        self.observation_interface.inject(
            gamestate,
            self.input_dict
        )
        self.metrics.update(gamestate)
        reward = self.reward_function.compute_step_rewards(gamestate)

        self.step_count += 1
        done = self.step_count >= self.episode_length
        dones = {
            "__all__": done,
            0: done
        }
        if done:
            self.on_episode_end()


        return {0: self.input_dict}, {0: reward}, dones, dones, self.empty_info_dict


    def on_episode_end(self):
        self.console.terminate_video()

    def get_episode_metrics(self) -> dict:
        d = {
        }
        d.update(self.reward_function.get_metrics())
        d.update(self.metrics.get_metrics(self.console.get_gamestate()))
        return d








