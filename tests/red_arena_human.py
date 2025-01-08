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
    envs = [PolarisRedArena(
        env_index=i,
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

    ) for i in range(1)
    ]
    env0 = envs[0]


    for env in envs:
        env.reset()

    action = None

    for i in range(500):
        if not env0.input_interface.human_inputs:
            action = env0.action_space.sample()
            time.sleep(0.25)

        env_outputs = []
        for env in envs:
            observations, rewards, _, _, _ = env.step({0: action})
            env_outputs.append((observations, rewards, env.console.get_gamestate()))


        for env in envs:
            gs = env.console.get_gamestate()
            # print(gs.party_moves, gs.party_pps, gs.party_types, gs.party_attributes)
            # print(gs.sent_out_party_moves, gs.sent_out_party_pps, gs.sent_out_party_types, gs.sent_out_party_attributes)
            # print(gs.sent_out_opponent_moves, gs.sent_out_opponent_pps, gs.sent_out_opponent_types, gs.sent_out_opponent_attributes)




if __name__ == '__main__':
    run()