import time
from typing import Tuple

from deepred.polaris_env.red_arena.polaris_red_arena import PolarisRedArena



def run(
        game_path: str = "faster_red6.gbc",
        episode_length: int = 500,
        human_inputs: bool = True,
        downscaled_screen_shape: tuple = (72, 80),
        framestack: int = 1,
        stack_oldest_only: bool = False,
        enabled_patches: Tuple[str] = ("out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
                                    "victory_road", "elevator", "freshwater_trade", "seafoam_island"),
        reward_scales=dict(win=1),
        session_path: str = "red_arena_tests",
        speed_limit: int = 1,
        render: bool = True,

        wild_battle_savestate: str = "wild_battle.state",
        trainer_battle_savestate: str = "trainer_battle.state",
        level_mean_bounds: Tuple[int, int] = (5, 60),
        party_level_std_max: int = 10,
        opponent_level_std_max: int = 3,
        wild_battle_chance: float = 1.,
):
    env = PolarisRedArena(
        env_index=0,
        game_path=game_path,
        episode_length=episode_length,
        human_inputs=human_inputs,
        downscaled_screen_shape=downscaled_screen_shape,
        framestack=framestack,
        stack_oldest_only=stack_oldest_only,
        reward_scales=reward_scales,
        enabled_patches=enabled_patches,
        session_path=session_path,
        speed_limit=speed_limit,
        render=render,

        wild_battle_savestate=wild_battle_savestate,
        trainer_battle_savestate=trainer_battle_savestate,
        level_mean_bounds=level_mean_bounds,
        party_level_std_max=party_level_std_max,
        opponent_level_std_max=opponent_level_std_max,
        wild_battle_chance=wild_battle_chance

    )


    env.reset()
    action = None

    for i in range(500):
        if not env.input_interface.human_inputs:
            action = env.action_space.sample()
            time.sleep(0.25)

        observations, rewards, _, _, _ = env.step({0: action})





if __name__ == '__main__':
    run()