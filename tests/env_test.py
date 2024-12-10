from typing import Tuple

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_env.rewards import Goals
from deepred.polaris_utils.counting import HashScales



def run(
     game_path: str = "faster_red6.gbc",
     episode_length: int = 2048,
     human_inputs: bool = True,
     downscaled_screen_shape: tuple = (36, 40),
     framestack: int = 3,
     stack_oldest_only: bool = False,
     map_history_length: int = 10,
     flag_history_length: int = 10,
     enabled_patches: Tuple[str] = ("out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
                                    "victory_road", "elevator", "freshwater_trade", "seafoam_island"),
     reward_scales: dict = dict(exploration=1),
     savestate: str | None = "faster_red_post_parcel_pokeballs.state",
     session_path: str = "red_tests",
     record: bool = False,
     speed_limit: int = 1,
     record_skipped_frame: bool = False,
):
    envs = [PolarisRed(
        env_index=i,
        game_path=game_path,
        episode_length=episode_length,
        human_inputs=human_inputs,
        downscaled_screen_shape=downscaled_screen_shape,
        framestack=framestack,
        stack_oldest_only=stack_oldest_only,
        reward_scales=reward_scales,
        savestate=savestate,
        map_history_length=map_history_length,
        flag_history_length=flag_history_length,
        enabled_patches=enabled_patches,
        session_path=session_path,
        render=True,
        record=record,
        speed_limit=speed_limit,
        record_skipped_frame=record_skipped_frame
    ) for i in range(1)
    ]
    options = {0: {"count_based_exploration_scales": HashScales({})}}
    for env in envs:
        env.reset(options=options)
    env0 = envs[0]

    action = None

    for i in range(2048):
        if not env0.input_interface.human_inputs:
            action = env0.action_space.sample()
        env_outputs = []
        for env in envs:
            observations, rewards, _, _, _ = env.step({0: action})
            env_outputs.append((observations, rewards, env.console.get_gamestate()))

        for env in envs:
            print(env.console.get_gamestate().frame)



if __name__ == '__main__':
    run()