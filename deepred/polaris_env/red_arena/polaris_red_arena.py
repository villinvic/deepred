from pathlib import Path
from typing import Tuple, Union, Dict, Any, SupportsFloat

from gymnasium.core import ObsType, ActType
from polaris.environments import PolarisEnv

from deepred.polaris_env.action_space import PolarisRedActionSpace
from deepred.polaris_env.gb_console import GBConsole
from deepred.polaris_env.pokemon_red.enums import RamLocation
from deepred.polaris_env.red_arena.observation_space import PolarisRedArenaObservationSpace
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
    -> on the first button press, the battle starts
    -> episode ends when the battle is finished, or when maximum number of steps is reached



    ----------
    For later:
    -> Implement a specific trainer (arena_trainer.py) (Victor) to train agents on this environment.
    Final Goals:
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

        self.observation_interface = PolarisRedArenaObservationSpace(
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
        self.done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Called each time we want to initialise the environment for a new episode

        TODO
        """
        sampled_battle = self.battle_sampler()
        initial_gamestate = self.console.reset(sampled_battle)

        # add a hook to the console tick function
        # so that we update the ram each frame before the battle begins
        setattr(self.console, "old_tick", self.console.tick)

        ram_to_observe = [
            RamLocation.ENEMY_POKEMON_SPECIES,
            RamLocation.ENEMY_POKEMON_ID,
            RamLocation.ENEMY_POKEMON_TYPE0,
            RamLocation.ENEMY_POKEMON_TYPE1,
            RamLocation.ENEMY_POKEMON_MOVE0,
            RamLocation.ENEMY_POKEMON_MOVE1,
        ]
        self.reward_function = PolarisRedArenaRewardFunction(
            reward_scales=self.reward_scales,
            inital_gamestate=initial_gamestate,
        )
        def print_ram_values():
            for addr in ram_to_observe:
                print(f"{addr.name:<30}: {self.console.memory[addr]}")

        def hook(count, render):
            print("-----RAM BEFORE GAME UPDATE-----")
            print_ram_values()
            gs = self.console.old_tick(count, render)
            print("-----RAM AFTER GAME UPDATE -----")
            print_ram_values()


            if not gs.is_in_battle:
                sampled_battle.inject_to_ram(self.console.memory)
            elif not self.done:
                print("done")
                self.done = True
                sampled_battle.inject_to_ram(self.console.memory)
            #sampled_battle.inject_to_ram(self.console.memory)

            return gs

        setattr(self.console, "tick", hook)


        self.input_dict = self.observation_space.sample()
        self.observation_interface.inject(
            initial_gamestate,
            self.input_dict
        )
        self.step_count = 0
        # c.f. polaris_red.py to update the observations.
        return {0: self.input_dict}, self.empty_info_dict

    def step(
        self,
        action_dict
    ):
        event = self.input_interface.get_event(action_dict[0], self.console.get_gamestate())

        gamestate = self.console.process_event(event)

        rewards = self.reward_function.compute_step_rewards(gamestate)
        self.observation_interface.inject(
            gamestate,
            self.input_dict
        )

        # Will see if is_lazy can be useful to be True
        """early_termination = self.reward_function.is_lazy()
        if early_termination:
            rewards -= self.reward_scales["early_termination"]"""

        self.step_count += 1
        done = self.step_count >= self.episode_length# or early_termination
        dones = {
            "__all__": done,
            0: done,
        }

        if done or gamestate._additional_memory.battle_staling_checker.is_battle_staling():
            self.on_episode_end()

        # you should only modify how we get observations, rewards and dones
        return {0: self.input_dict}, rewards, dones, dones, self.empty_info_dict

    def on_episode_end(self):
        self.console.terminate_video()

    def get_episode_metrics(self) -> dict:
        """
        TODO
        :return: metrics for the episode (reported to wandb)
        """

        metrics = {}

        return metrics