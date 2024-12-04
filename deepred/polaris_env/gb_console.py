from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple

import mediapy
import numpy as np
from gymnasium.error import ResetNeeded
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from deepred.polaris_env.action_space import PolarisRedActionSpace
from deepred.polaris_env.additional_memory import AdditionalMemory
from deepred.polaris_env.enums import StartMenuItem, EventFlag, BagItem, RamLocation
from deepred.polaris_env.game_patching import AgentHelper, GamePatching
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.observation_space import PolarisRedObservationSpace

RELEASE_EVENTS = {
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
    WindowEvent.PASS: None
}


class GBConsole(PyBoy):


    def __init__(
            self,
            console_id: int,
            game_path: str,
            render: bool = False,
            speed_limit: int = 1,
            record: bool = False,
            record_skipped_frames: bool = False,
            output_dir: Path = Path("deepred_console"),
            savestate: Union[None, str] = None,
            enabled_patches: Tuple[str] = (),
            **kwargs
    ):
        """
        This wraps the PyBoy class for our convenience.

        :param game_path: Path to the gb/gba rom.
        :param render: Enables rendering of the game for visualisation. Otherwise, the game runs headlessly and without
            speed limit.
        :param render_skipped_frame: If True, we render all frames, even the "skipped" ones
        :param output_dir: Output directory for the console: screenshots, videos, dumps, etc.
        :param savestate: If None, loads the game normally, otherwise, boots the game starting from the given state.
        :param kwargs: Additional parameters to pass to the PyBoy constructor.
        :param patch: patches the game to make it more RL-friendly.
        """
        self.console_id = console_id

        super().__init__(
            game_path,
            window='SDL2' if render else 'null',
            **kwargs
        )
        self._ram_helper = self.game_wrapper()

        self.game_patcher = GamePatching(enabled_patches)
        self.agent_helper = AgentHelper()

        self.render = render
        self.min_frame_skip = 20
        self.set_emulation_speed(speed_limit if render else 0)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.record_skipped_frames = record_skipped_frames
        self.record = record

        self.invalid_count = 0
        self.init_paths(output_dir)

        self._savestate = savestate
        self._frame = 0
        self._step = 0
        self._additional_memory : Union[None, AdditionalMemory] = None
        self._gamestate = GameState(self)

    def init_paths(
            self,
            output_dir: Path
    ):
        """
        :param output_dir: dir where to init all the paths.
        :return:
        """
        self.dump_dir = output_dir / Path('dumps')
        self.dump_dir.mkdir(exist_ok=True)

        self.dump_index = 0
        if self.record:
            self.video_dir = output_dir / Path('recordings')
            self.video_dir.mkdir(exist_ok=True)
            self.video_index = 0

    def get_gamestate(self) -> GameState:
        return self._gamestate

    def reset(self) -> GameState:
        with open(self._savestate, "rb") as f:
            self.load_state(f)
        if self.record:
            self.initialise_video()

        self._frame = 0
        self._step = 0
        self._additional_memory : Union[None, AdditionalMemory] = None
        self._gamestate = self.tick(2)
        return self._gamestate

    def initialise_video(self):
        path = self.video_dir / Path(f'episode_{self.video_index}').with_suffix('.mp4')
        while path.exists():
            self.video_index += 1
            path = self.video_dir / Path(f'episode_{self.video_index}').with_suffix('.mp4')

        self.recorder = mediapy.VideoWriter(path, (144, 160), fps=20)
        self.recorder.__enter__()

    def terminate_video(self):
        if self.record:
            self.recorder.close()

    def step_event(
            self,
            event: WindowEvent
    ) -> GameState:
        """
        Sends the event to the console, and steps the console until the player is actionable once more.
        :param event: event to send to the console.
        :return: The next gamestate.
        """

        # validate input first
        if not self.validate_input(event):
            self.invalid_count += 1
            if self.invalid_count > 200:
                self.handle_error("We are stuck somewhere with invalid inputs.")
            return self._gamestate
        self.invalid_count = 0


        # In order for actions to be stateless, we need to send it for 8 frames.
        self.send_input(event)
        self.tick(8)
        if event != WindowEvent.PASS:
            self.send_input(RELEASE_EVENTS[event])

        # Skip frames until we are actionable.
        try:
            return self.skip_frames()
        except RecursionError as e:
            self.handle_error("Stuck stepping the console.")

    def tick(self, count=1, render=True) -> GameState:
        super().tick(count, render)
        if self.record and render:
            self.add_video_frame()

        self._frame += count

        return GameState(self)

    def add_video_frame(self):
        self.recorder.add_image(self._gamestate.screen)

    def skip_frames(self) -> GameState:
        """
        Skips the game until we know the player can act.
        min_frame_skip is to ensure we can have an actionable action after walking.
        :return: The last gamestate
        """
        total_frames_ticked = 0
        skip_count = 1 if self.record else 4

        # we already ticked 8 frames
        frames_to_skip =  self.min_frame_skip - 1 - 8
        additional_skip = 0
        while total_frames_ticked <= frames_to_skip:
            gamestate = self.tick(skip_count, self.record_skipped_frames or self.render)
            total_frames_ticked += skip_count
            if (
                    self._gamestate.map != gamestate.map
                or
                    self._gamestate.is_in_battle and not gamestate.is_in_battle
            ):
                frames_to_skip += 33

            # map = gamestate.map
            # in_battle = gamestate.is_in_battle
            self._gamestate = gamestate
            if self._gamestate.is_skippable_frame():
                total_frames_ticked -= skip_count * 0.75
                additional_skip += 1
                if additional_skip > 500:
                    self.handle_error("stuck skipping.")

        self._step += 1
        self._gamestate = self.tick(1, render=True)

        # Is this fine to patch here ?
        self.game_patcher.patch(self._gamestate)

        return self._gamestate

    def validate_input(
            self,
            event: WindowEvent
    ) -> bool:
        """
        We need to validate inputs sometimes, to prevent the bot from doing things we don't want, such as
        changing the options.
        :param event: Input to send to the console
        :return: True if the input is validated
        """
        if self._gamestate.start_menu_item in (StartMenuItem.POKEDEX, StartMenuItem.SAVE, StartMenuItem.TRAINER,
                                               StartMenuItem.OPTION) and event == WindowEvent.PRESS_BUTTON_A:
            return False

        # ...

        return True

    def set_instant_speed(self):
        if not self._gamestate.instant_text:
            self.memory[RamLocation.wd730] = self.memory[RamLocation.wd730] | (1<<6)

    def handle_error(
            self,
            msg: str
    ):
        """
        :param msg: message describing the error.
        """
        self.dump_gamestate(msg)
        raise ResetNeeded(msg)

    def dump_gamestate(
            self,
            msg: str
    ):
        """
        :param msg: message to store into the dumped file.
        """
        path = self.dump_dir / Path(f'dump_num_{self.dump_index}').with_suffix('.log')
        while path.exists():
            self.dump_index += 1
            path = self.dump_dir / Path(f'dump_num_{self.dump_index}').with_suffix('.log')
        self._gamestate.dump(self.dump_dir / Path(f'dump_num_{self.dump_index}'), msg)


