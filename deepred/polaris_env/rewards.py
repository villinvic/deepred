from dataclasses import asdict
from typing import NamedTuple, Dict

import numpy as np
import tree
from attr import dataclass

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_utils.counting import HashScales, hash_function


# self.ball_price_to_item_value = {
#     200: 1, # pokeball
#     600: 2, # greatball
#     1200: 1.5, # ultraball
#     0: 2, # ball was obtained
# }
#
# self.heal_price_to_item_value = {
#     3000 : 4,  # full restore
#     2500: 4,  # max potion
#     1500 : 3,  # hyper potion
#     1200: 1.5,  # ultraball
#     0   : 5,  # ball was obtained
# }

# self.reward_function_config = {
#     BLACKOUT: - 0.05,
#     SEEN_POKEMONS: 0.3,
#     TOTAL_EXPERIENCE: 20.,  # 0.5
#     BADGE_SUM: 100.,
#     MAPS_VISITED: 0.2,  # 3.
#     TOTAL_EVENTS_TRIGGERED: 0.06,  # TODO : bugged
#     MONEY: 10.,
    # COORDINATES              :   - 5e-4,
    # COORDINATES + "_NEG"     :   0.003 * 0.9,
    # COORDINATES + "_POS"     :   0.003,
    #PARTY_HEALTH: 3.,

    # GOAL_TASK                :  0.5,

    # ITEMS                    :  0.1,


class Goals(NamedTuple):
    """
    Rewards collected by the agent each step.
    The remaining rewards will be on the Polaris side
    (we need to communicate visitation stats between the learner and the workers).
    """
    seen_pokemons: float = 0
    badges: float = 0
    experience: float = 0

    # computed with map-(event flags) hash
    exploration: float = 0


def _get_goals_delta(
        previous: Goals,
        current: Goals
) -> Goals:
    return tree.map_structure(
        lambda p, c: c - p,
    previous, current
    )

def accumulate_goal_stats(
        new: Goals,
        total: Goals
) -> Goals:
    return tree.map_structure(
        lambda n, t: n + t,
    new, total
    )

def _compute_step_reward(
        rewards: Goals,
        scales: Goals
) -> float:

    return sum([*tree.map_structure(
        lambda r, s: r * s,
    rewards, scales
    )])

class PolarisRedRewardFunction:
    def __init__(
            self,
            reward_scales: dict | None,
            count_based_exploration_scales: HashScales,
            inital_gamestate: GameState
    ):
        """
        This class takes care of computing rewards.
        :param reward_scales: scales for each goal.
        :param inital_gamestate: initial state of the game to setup the reward function.
        """
        self.episode_max_party_exp = -np.inf
        self.episode_max_level = -np.inf

        self.scales = Goals() if reward_scales is None else Goals(** reward_scales)
        self.delta_goals = Goals()

        init_hash = hash_function((inital_gamestate.map, inital_gamestate.event_flags))
        self.total_exploration = 0
        self.visited_hash = {init_hash}

        self._cumulated_rewards = Goals()
        self._previous_goals = self._extract_goals(inital_gamestate)
        self.count_based_exploration_scales = count_based_exploration_scales



    def _extract_goals(
            self,
            gamestate: GameState
    ) -> Goals:
        #     seen_pokemons: float = 0
        seen_pokemons = gamestate.species_seen_count
        #     badges: float = 0
        badges = sum(gamestate.badges)
        #     experience: float = 0
        experience = sum(gamestate.party_experience)
        if experience > self.episode_max_party_exp:
            self.episode_max_party_exp = experience

        max_level = max(gamestate.party_level)
        if max_level > self.episode_max_level:
            # The player could always move its higher leveled pokemon into the pc
            # This may be a breach to hack the experience related rewards.
            # So we keep the episode maximum level to prevent that.
            self.episode_max_level = max_level

        map_event_flag_hash = hash_function((gamestate.map, gamestate.event_flags))
        if map_event_flag_hash not in self.visited_hash:
            self.total_exploration += self.count_based_exploration_scales[map_event_flag_hash]
            self.visited_hash.add(map_event_flag_hash)

        return Goals(
            seen_pokemons=seen_pokemons,
            badges=badges,
            experience=self.episode_max_party_exp,
            exploration=self.total_exploration
        )

    def _get_goal_updates(
            self,
            goals: Goals
    ) -> Goals:
        return _get_goals_delta(self._previous_goals, goals)

    def compute_step_rewards(
            self,
            gamestate: GameState
    ) -> float:
        goals = self._extract_goals(gamestate)
        goal_updates = self._get_goal_updates(goals)

        rewards = Goals(
            seen_pokemons=goal_updates.seen_pokemons,
            badges=goal_updates.badges,
            experience=goal_updates.experience / self.episode_max_level**3,
            exploration=goal_updates.exploration
        )
        self._cumulated_rewards = accumulate_goal_stats(rewards, self._cumulated_rewards)
        self._previous_goals = goals

        return _compute_step_reward(rewards, self.scales)

    def get_metrics(self) -> dict:
        return {
            "rewards": self._cumulated_rewards,
            "episode_max_level": self.episode_max_level
        }







