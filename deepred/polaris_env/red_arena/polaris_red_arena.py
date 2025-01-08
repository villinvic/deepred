from pathlib import Path
from typing import Tuple, Union, Dict, Any, SupportsFloat

from gymnasium.core import ObsType, ActType
from polaris.environments import PolarisEnv

from deepred.polaris_env.action_space import PolarisRedActionSpace
from deepred.polaris_env.gb_console import GBConsole
from deepred.polaris_env.observation_space import PolarisRedObservationSpace
from deepred.polaris_env.pokemon_red.enums import RamLocation
from deepred.polaris_env.red_arena.battle_sampler import BattleSampler
from deepred.polaris_env.red_arena.rewards import PolarisRedArenaRewardFunction


class PolarisRedArena(PolarisEnv):
    """
    Main environment class for the sub polaris red battling environment

    Environment focusing on battling in pokemon red

    Uses:
    -> find the best setup for learning proper battling

    Functionality:
    -> resets the game on battle startup
    -> can roll party as much as desired on startup
    -> on the first a press, the battle starts
    -> episode ends when the battle is won, or when maximum number of steps is reached


    TODO:
    -> implement all functions
    -> generate a set of savestates for battling (only Trainer Battles, no wild battles)
        -> must start the game with a battle about to start
        -> randomly sample a battle injected to ram (battle_sampler)

    -> Implement a specific trainer (arena_trainer.py) (Victor) to train agents on this environment.

    Later:
    -> mix Wild pokemon battles with Trainer Battles in order to learn when to
        -> catch
        -> get exp
        -> run away



    Goals:
    -> test what kind of neural network works best for battling in pokemon red
    -> get an agent that learns to:
        -> win battles
        -> level up its lower leveled pokemons
        -> other ?

        Thus:
            -> learns to pick proper moves
            -> learns to swap pokemons
            -> learns to lead with the right pokemon at the beginning of a battle
    """

    env_id = "PolarisRedArena"


    def __init__(
            self,
            env_index=-1,
            game_path: str = "faster_red.gbc",
            episode_length=500,
            human_inputs: bool = False,
            downscaled_screen_shape: Tuple = (72, 80),
            framestack: int = 1,
            stack_oldest_only: bool = False,
            reward_scales: dict | None = None,
            wild_battle_savestate: str = "wild_battle.state",
            trainer_battle_savestate: str = "trainer_battle.state",
            level_mean_bounds: Tuple[int, int] = (5, 60),
            party_level_std_max: int = 10,
            opponent_level_std_max: int = 3,
            wild_battle_chance: float = 0.5,
            enabled_patches: Tuple[str] = (),
            session_path: str = "red_arena_tests",
            render: bool = True,
            speed_limit: int = 1,
            **config
    ):
        super().__init__(env_index, **config)
        self._agent_ids = {0}
        self.empty_info_dict = {0: {}}

        self.render = env_index == 0 and render
        self.session_path = Path(session_path)
        self.step_count = 0
        self.episode_length = episode_length

        self.session_path.mkdir(exist_ok=True)

        self.battle_sampler = BattleSampler(
            wild_battle_savestate=wild_battle_savestate,
            trainer_battle_savestate=trainer_battle_savestate,
            level_mean_bounds=level_mean_bounds,
            party_level_std_max=party_level_std_max,
            opponent_level_std_max=opponent_level_std_max,
            wild_battle_chance=wild_battle_chance
        )

        # We need to setup the console here.
        self.console = GBConsole(
            console_id=self.env_index,
            game_path=game_path,
            render=self.render,
            speed_limit=speed_limit,
            record=False,
            record_skipped_frames=False,
            output_dir=self.session_path / Path(f"console_{self.env_index}"),
            default_savestate=None,
            map_history_length=1,
            flag_history_length=1,
            enabled_patches=enabled_patches,
            checkpoint_identifiers=None,
            **config
        )
        # We perform a reset + tick to get the gamestate.
        sampled_battle = self.battle_sampler()
        self.console.reset(sampled_battle)

        self.observation_interface = PolarisRedObservationSpace(
            downscaled_screen_shape=downscaled_screen_shape,
            framestack=framestack,
            stack_oldest_only=stack_oldest_only,
            dummy_gamestate=self.console.tick()
        )


        self.input_interface = PolarisRedActionSpace(
            enable_start=False,
            enable_pass=False,
            enable_roll_party=True,
            human_inputs=human_inputs,
        )

        self.action_space = self.input_interface.gym_spec
        self.observation_space = self.observation_interface.gym_spec


        self.reward_function: Union[PolarisRedArenaRewardFunction, None] = None
        self.input_dict: Union[Dict, None] = None

        self.reward_scales = reward_scales


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Called each time we want to initialise the environment for a new episode

        ->
        """
        sampled_battle = self.battle_sampler()
        gamestate = self.console.reset(sampled_battle)

        # add a hook to the tick function so that we ensure we have the right pokemons
        setattr(self.console, "old_tick", self.console.tick)
        def hook(count, render):
            gs = self.console.old_tick(count, render)
            sampled_battle.inject_to_ram(self.console.memory)
            return gs

        setattr(self.console, "tick", hook)


        self.input_dict = self.observation_space.sample()
        self.observation_interface.inject(
            gamestate,
            self.input_dict
        )
        self.step_count = 0

        return {0: self.input_dict}, self.empty_info_dict



    def step(
        self,
        action_dict
    ):
        action = action_dict[0]

        event = self.input_interface.get_event(action, self.console.get_gamestate())
        self.console.process_event(event)

        done = {
            "__all__": False,
            0: False,
        }

        return {0: self.input_dict}, {0: 0}, done, done, self.empty_info_dict



    def get_episode_metrics(self) -> dict:
        """
        :return: metrics for the episode (reported to wandb)
        """

        metrics = {}

        # TODO


        return metrics