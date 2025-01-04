from functools import partial
from pathlib import Path
from typing import Union, Tuple, Callable
import mediapy
from gymnasium.error import ResetNeeded
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from deepred.polaris_env.action_space import CustomEvent
from deepred.polaris_env.additional_memory import AdditionalMemory
from deepred.polaris_env.battle_parser import parse_battle_state, BattleState
from deepred.polaris_env.env_checkpointing.env_checkpoint import EnvCheckpoint
from deepred.polaris_env.env_checkpointing.env_checkpointer import EnvCheckpointer
from deepred.polaris_env.pokemon_red.enums import StartMenuItem, RamLocation
from deepred.polaris_env.game_patching import GamePatching
from deepred.polaris_env.agent_helper import AgentHelper
from deepred.polaris_env.gamestate import GameState

RELEASE_EVENTS = {
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
    WindowEvent.PASS: WindowEvent.PASS
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
            default_savestate: Union[None, str] = None,
            map_history_length: int = 10,
            flag_history_length: int = 10,
            enabled_patches: Tuple[str] = (),
            checkpoint_identifiers: Tuple[str] = (),
            max_num_savestates_per_checkpoint: int = 15,

            **kwargs
    ):
        """
        This wraps the PyBoy class for our convenience.

        :param game_path: Path to the gb/gba rom.
        :param render: Enables rendering of the game for visualisation. Otherwise, the game runs headlessly and without
            speed limit.
        :param render_skipped_frame: If True, we render all frames, even the "skipped" ones
        :param output_dir: Output directory for the console: screenshots, videos, dumps, etc.
        :param default_savestate: If None, loads the game normally, otherwise, boots the game starting from the given state.
        :param enabled_patches: patches to enable for the game to be more RL-friendly.
        :param checkpoint_identifiers: gamestate attributes to use to save checkpoints.
        :param max_num_savestates_per_checkpoint: numbers of checkpoints to keep per checkpoint id.

        :param kwargs: Additional parameters to pass to the PyBoy constructor.
        """
        self.console_id = console_id

        super().__init__(
            game_path,
            window='SDL2' if render else 'null',
            **kwargs
        )
        self._ram_helper = self.game_wrapper

        self.game_patcher = GamePatching(enabled_patches)
        self.agent_helper = AgentHelper()

        self.render = render
        self.min_frame_skip = 20  # 24 (seems unnecessarily large ?)
        self.set_emulation_speed(speed_limit if render else 0)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.record_skipped_frames = record_skipped_frames
        self.record = record

        self.invalid_count = 0
        self.init_paths(output_dir)

        self._default_savestate = default_savestate
        self._frame = 0
        self._step = 0
        self._additional_memory_maker = partial(
            AdditionalMemory,
            map_history_length=map_history_length,
            flag_history_length=flag_history_length
        )

        self.checkpoint_identifiers = checkpoint_identifiers
        self.max_num_savestates_per_checkpoint = max_num_savestates_per_checkpoint
        self._checkpointer = EnvCheckpointer(
            self.output_dir.parent,
            self.checkpoint_identifiers,
            self.max_num_savestates_per_checkpoint
        )

        self._additional_memory = self._additional_memory_maker()
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

    def reset(
            self,
            ckpt: EnvCheckpoint = EnvCheckpoint()
    ) -> GameState:
        """
        :param ckpt: Checkpoint to load for resetting (gamestate and savestate).
        :return: The initial gamestate.
        """
        if ckpt.savestate is None:
            # restore with default savestate if empty checkpoint given
            with open(self._default_savestate, "rb") as f:
                self.load_state(f)
            self._additional_memory = self._additional_memory_maker()
            self._frame = 0
            self._step = 0
        else:
            self.load_state(ckpt.savestate)
            self._additional_memory = ckpt.additional_memory
            self._frame = ckpt.frame
            self._step = ckpt.step

        if self.record:
            self.initialise_video()

        self._gamestate = self.tick(2)
        self.game_patcher.patch(self._gamestate)
        self._additional_memory.update(self._gamestate)
        self._checkpointer = EnvCheckpointer(
            self.output_dir.parent,
            self.checkpoint_identifiers,
            self.max_num_checkpoints,
            ckpt.ckpt_id,
        )

        return self._gamestate

    def initialise_video(self):
        path = self.video_dir / Path(f'episode_{self.video_index}').with_suffix('.mp4')
        while path.exists():
            self.video_index += 1
            path = self.video_dir / Path(f'episode_{self.video_index}').with_suffix('.mp4')

        self.recorder = mediapy.VideoWriter(path, (144, 160), fps=180)
        self.recorder.__enter__()

    def terminate_video(self):
        if self.record:
            self.recorder.close()

    def step_event(
            self,
            event: WindowEvent
    ):
        """
        Sends the event to the console, and steps the console until the player is actionable once more.
        :param event: event to send to the console.
        :return: The next gamestate.
        """

        # In order for actions to be stateless, we need to send it for 8 frames.
        self.send_input(event)
        self.tick(8, render=self.render and self.record_skipped_frames)
        self.send_input(RELEASE_EVENTS[event])
        # Skip frames until we are actionable.
        try:
            self.skip_frames()
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

    def skip_frames(self):
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
                    self._gamestate.map != gamestate.map # map loading screen
                or
                    self._gamestate.is_in_battle and not gamestate.is_in_battle # battle transition screen
            ):
                frames_to_skip += 38

            self._gamestate = gamestate
            if self._gamestate.is_skippable_frame():
                total_frames_ticked -= skip_count * 0.99 # make sure we do not skip forever.
                additional_skip += 1
                if additional_skip > 500:
                    self.handle_error("stuck skipping.")

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


    def step_force_ram(
            self,
            ram_modifier: Callable,
            num_frames=20,
            event=WindowEvent.PASS
    ):
        """
        Utility function to run the game forcing some ram values.

        """
        self.send_input(event)
        for i in range(num_frames):
            if i == 8:
                self.send_input(RELEASE_EVENTS[event])

            gamestate = self.tick(1, render=self.render or self.record_skipped_frames)
            ram_modifier(gamestate._ram)


    def handle_world_event(
            self,
            event: WindowEvent
    ) -> bool:
        """
        Handles bot inputs when not in battle.
        Returns if we are waiting for an input of the bot.
        """
        finalise = True

        if self._gamestate.skippable_with_a_press:
            self.step_event(WindowEvent.PRESS_BUTTON_A)
            return False

        if event == WindowEvent.PRESS_BUTTON_A and not self._gamestate.open_menu:
            # We should not be able to open any other menu beside the shop menu if we disable the START menu
            # but ok .
            if self._gamestate.is_at_pokemart and self._gamestate.mart_items:
                finalise = not self.agent_helper.shopping(self._gamestate)
            elif self._gamestate.can_use_pc:
                finalise = not self.agent_helper.manage_party(self._gamestate)
            else:
                finalise = not self.agent_helper.field_move(self._gamestate, self.step_force_ram)

        return finalise

    def handle_battle_event(
            self,
            event: WindowEvent
    ) -> bool:
        """
        Handles bot inputs when in battle.
        Returns if we are waiting for an input of the bot.
        """
        c = 0
        finalise = True
        while True:
            c += 1
            battle_state = parse_battle_state(self._gamestate)
            if battle_state in (BattleState.NOT_IN_BATTLE, BattleState.ACTIONABLE):
                break
            elif battle_state == BattleState.SKIPPABLE:
                self.step_event(WindowEvent.PASS)
                finalise = False

            elif battle_state == BattleState.OTHER:
                self.step_event(WindowEvent.PRESS_BUTTON_A)
                finalise = False
            elif battle_state == BattleState.SWITCH:
                self.agent_helper.switch(self._gamestate)
                self.step_event(WindowEvent.PRESS_BUTTON_A)
                finalise = False
            elif battle_state == BattleState.LEARN_MOVE:
                # force to learn the move
                self.step_event(WindowEvent.PRESS_BUTTON_A)
                finalise = False
            elif battle_state == BattleState.REPLACE_MOVE:
                should_learn_move = self.agent_helper.should_learn_move(self._gamestate)
                if should_learn_move:
                    self.step_event(WindowEvent.PRESS_BUTTON_A)
                else:
                    self.step_event(WindowEvent.PRESS_BUTTON_B)
                finalise = False
            elif battle_state == BattleState.ABANDON_MOVE:
                # force to abandon the move
                self.step_event(WindowEvent.PRESS_BUTTON_A)
                finalise = False
            elif battle_state == BattleState.NICKNAME:
                # auto decline nickname
                self.step_event(WindowEvent.PRESS_BUTTON_B)
                finalise = False

            if c >= 8:
                # break after some frames anyways
                break
        return finalise

    def process_event(
            self,
            event: WindowEvent | CustomEvent
    ) -> GameState:
        self._checkpointer.do_checkpoint_if_needed(
            self.save_state,
            self._gamestate
        )
        try:
            if event == CustomEvent.ROLL_PARTY:
                self.agent_helper.roll_party(gamestate=self._gamestate)
                return self.get_actionable_frame()

            if self._gamestate.is_in_battle:
                finalise = self.handle_battle_event(event)
            else:
                finalise = self.handle_world_event(event)

            if finalise:
                self.step_event(event)

            return self.get_actionable_frame()
        except Exception as e:
            self.handle_error(str(e))



    def get_actionable_frame(self) -> GameState:

        self._gamestate = self.tick(1, render=True)
        # Is this fine to patch here ?
        self.game_patcher.patch(self._gamestate)
        self._additional_memory.update(self._gamestate)
        self._step += 1
        return self._gamestate

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


