
from typing import Union, Dict, Optional, Tuple


from pathlib import Path

from polaris.environments import PolarisEnv
from pyboy.utils import WindowEvent

from deepred.polaris_env.action_space import PolarisRedActionSpace, CustomEvent
from deepred.polaris_env.gb_console import GBConsole
from deepred.polaris_env.metrics import PolarisRedMetrics
from deepred.polaris_env.observation_space import PolarisRedObservationSpace
from deepred.polaris_env.rewards import PolarisRedRewardFunction
from deepred.polaris_env.streaming import BotStreamer


class PolarisRed(PolarisEnv):
    env_id = "PolarisRed"

    # This environment was declined from
    # https://github.com/CJBoey/PokemonRedExperiments1/blob/master/baselines/boey_baselines2/red_gym_env.py#L2373
    # Although this is not a fork of their work, we employed many of their ideas.

    def __init__(
            self,
            env_index=-1,
            game_path: str= "faster_red.gbc",
            episode_length=2048, # 2048 is short.
            enable_start: bool = False,
            enable_pass: bool = False,
            enable_roll_party: bool = True,
            human_inputs: bool = False,
            downscaled_screen_shape: Tuple = (36, 40),
            framestack: int = 3,
            stack_oldest_only: bool = False,
            reward_scales: dict  | None = None,
            reward_laziness_check_freq: int = 2048,
            reward_laziness_limit: int = 2048 * 4,
            savestate: Union[None, str] = None,
            map_history_length: int = 10,
            flag_history_length: int = 10,
            enabled_patches: Tuple[str] = (),
            session_path: str = "red_tests",
            render: bool = True,
            record: bool = False,
            speed_limit: int = 1,
            record_skipped_frame: bool= False,
            stream: bool = True,
            bot_name: str = "deepred",
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

        # We need to setup the console here.
        self.console = GBConsole(
            console_id=self.env_index,
            game_path=game_path,
            render=self.render,
            speed_limit=speed_limit,
            record=record,
            record_skipped_frames=record_skipped_frame,
            output_dir=self.session_path / Path(f"console_{self.env_index}"),
            savestate=savestate,
            map_history_length = map_history_length,
            flag_history_length = flag_history_length,
            enabled_patches=enabled_patches,
            **config
        )
        # We perform a reset + tick to get the gamestate.
        self.console.reset()

        self.observation_interface = PolarisRedObservationSpace(
            downscaled_screen_shape=downscaled_screen_shape,
            framestack=framestack,
            stack_oldest_only=stack_oldest_only,
            dummy_gamestate=self.console.tick()
        )

        self.input_interface = PolarisRedActionSpace(
            enable_start=enable_start,
            enable_pass=enable_pass,
            enable_roll_party=enable_roll_party,
            human_inputs=human_inputs,
        )

        self.action_space = self.input_interface.gym_spec
        self.observation_space = self.observation_interface.gym_spec


        self.stream = stream
        if stream:
            self.streamer = BotStreamer(self.env_index, bot_name=bot_name)

        self.metrics : Union[PolarisRedMetrics, None] = None
        self.reward_scales = reward_scales
        self.reward_laziness_check_freq = reward_laziness_check_freq
        self.reward_laziness = 0
        self.reward_laziness_limit = reward_laziness_limit
        self.reward_function: Union[PolarisRedRewardFunction, None] = None
        self.input_dict: Union[Dict, None] = None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: dict[str, dict] = None,
    ) -> Tuple[Dict[int, dict], dict]:

        gamestate = self.console.reset()
        self.metrics = PolarisRedMetrics()
        self.reward_function = PolarisRedRewardFunction(
            reward_scales=self.reward_scales,
            hash_counts=options[0]["hash_counts"],
            inital_gamestate=gamestate,
        )
        self.reward_laziness = 0

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
        if event == CustomEvent.DUMP_FRAME:
            path = self.session_path / Path("human_dump")
            path.mkdir(exist_ok=True)
            self.observation_interface.dump_observations(
                path / "observation",
                self.input_dict,
            )
            event = WindowEvent.PASS
        elif event == CustomEvent.SAVE_STATE:
            path = self.session_path / Path("human_dump")
            path.mkdir(exist_ok=True)

            with open(path / "save.state", "wb+") as f:
                self.console.save_state(f)
            event = WindowEvent.PASS

        gamestate = self.console.process_event(event)

        self.observation_interface.inject(
            gamestate,
            self.input_dict
        )
        self.metrics.update(gamestate)
        reward = self.reward_function.compute_step_rewards(gamestate)
        # TODO: put in config
        if reward <= 0.02:
            self.reward_laziness += 1
        else:
            self.reward_laziness = 0
        early_termination =(
                self.step_count % self.reward_laziness_check_freq == 0
                and self.reward_laziness >= self.reward_laziness_limit)

        if early_termination:
            reward -= self.reward_scales["early_termination"]

        self.step_count += 1
        # stop if reached step limit, or if we are stuck somewhere.
        done = self.step_count >= self.episode_length

        dones = {
            "__all__": done,
            0: done
        }
        if done:
            self.on_episode_end()

        if self.stream:
            self.streamer.send(self.console.get_gamestate())

        return {0: self.input_dict}, {0: reward}, dones, dones, self.empty_info_dict


    def on_episode_end(self):
        self.console.terminate_video()

    def get_episode_metrics(self) -> dict:
        d = {
        }
        d.update(self.reward_function.get_metrics())
        d.update(self.metrics.get_metrics(self.console.get_gamestate()))
        return d








