from deepred.env.enums import BagItem
from deepred.env.polaris_red import PolarisRed
from deepred.env.rewards import Goals
from deepred.polaris_utils.counting import HashScales

human_test = True

def run(
     game_path: str = "faster_red.gbc",
     episode_length: int = 2048,
     enable_start: bool = True,
     enable_pass: bool = human_test,
     downscaled_screen_shape: tuple = (72, 80),
     framestack: int = 3,
     stack_oldest_only: bool = False,
     observed_ram: tuple[str] = ("badges", "money"),
     observed_items: tuple[BagItem] = (BagItem.POKE_BALL, BagItem.POTION),
     reward_scales: Goals = Goals(seen_pokemons=0.1, experience=2, badges=5),
     savestate: str | None = "faster_red_post_parcel_pokeballs.state",
     session_path: str = "red_tests",
     record: bool = False,
     speed_limit: int = 1,
     record_skipped_frame: bool = False,
):
    envs = [PolarisRed(
        i,
        game_path,
        episode_length,
        enable_start,
        enable_pass,
        downscaled_screen_shape,
        framestack,
        stack_oldest_only,
        observed_ram,
        observed_items,
        reward_scales,
        savestate,
        session_path,
        True,
        record,
        speed_limit,
        record_skipped_frame
    ) for i in range(2)
    ]
    options = {0: {"count_based_exploration_scales": HashScales({})}}
    for env in envs:
        env.reset(options=options)
    env0 = envs[0]


    for i in range(2048):
        if human_test:
            action = env0.input_interface.human_input()
        else:
            action = env0.action_space.sample()
        env_outputs = []
        for env in envs:
            observations, rewards, _, _, _ = env.step({0: action})
            env_outputs.append((observations, rewards, env.console.get_gamestate()))

        output0 = env_outputs[0]
        print(i, output0[1])
        # for env_output in env_outputs[1:]:
        #     output_str = str(env_output)
        #     if output_str != output0:
        #         print("#"*20)
        #         print(output_str)
        #         print("#"*20)
        #         input()



if __name__ == '__main__':
    run()