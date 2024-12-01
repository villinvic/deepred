from ml_collections import ConfigDict

from deepred.env.enums import BagItem
from deepred.env.polaris_red import PolarisRed
from deepred.env.rewards import Goals
from deepred.models.impala_shallow import ImpalaShallowModel

def run(
     game_path: str = "faster_red.gbc",
     episode_length: int = 2048,
     enable_start: bool = True,
     enable_pass: bool = False,
     downscaled_screen_shape: tuple = (72, 80),
     framestack: int = 3,
     stack_oldest_only: bool = False,
     observed_ram: tuple[str] = ("badges", "money", "party_hp", "party_level", "sent_out", "in_battle", "bag_items",
                                 "event_flags"),
     observed_items: tuple[BagItem] = (BagItem.POKE_BALL, BagItem.GREAT_BALL, BagItem.ULTRA_BALL, BagItem.MASTER_BALL,
                                       BagItem.POTION, BagItem.SUPER_POTION, BagItem.HYPER_POTION, BagItem.FULL_RESTORE),
     reward_scales: Goals = Goals(seen_pokemons=1, experience=2, badges=10, exploration=1),
     savestate: str | None = "faster_red_post_parcel_pokeballs.state",
     session_path: str = "red_tests",
     record: bool = False,
     speed_limit: int = 1,
     record_skipped_frame: bool = False,
):
    env = PolarisRed(
        0,
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
        False,
        record,
        speed_limit,
        record_skipped_frame
    )

    model = ImpalaShallowModel(
        env.observation_space,
        env.action_space,
        ConfigDict({}),
    )
    model.setup()


if __name__ == '__main__':
    run()