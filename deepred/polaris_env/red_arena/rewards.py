from dataclasses import asdict
from typing import NamedTuple, Dict

import numpy as np
import tree
from attr import dataclass

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.enums import Map, FixedPokemonType
from deepred.polaris_utils.counting import HashCounts, hash_function


class ArenaGoals(NamedTuple):
    """
    ArenaGoals describing our performance in red arenaz
    """

    experience: float = 0
    """
    Rewards for leveling up/exp
    """

    win: float = 0
    """
    Rewards for winning the battle
    """

    fainting: float = 0
    """
    Penalty for having a pokemon fainting
    """

    timeout: float = 0
    """
    Penalty incurred when we reach episode termination
    """

    # Other rewards ...



def _get_goals_delta(
        previous: ArenaGoals,
        current: ArenaGoals
) -> ArenaGoals:
    return tree.map_structure(
        lambda p, c: c - p,
    previous, current
    )

def accumulate_goal_stats(
        new: ArenaGoals,
        total: ArenaGoals
) -> ArenaGoals:
    return tree.map_structure(
        lambda n, t: n + t,
    new, total
    )

def _compute_step_reward(
        rewards: ArenaGoals,
        scales: ArenaGoals
) -> float:

    return sum([*tree.map_structure(
        lambda r, s: r * s,
    rewards, scales
    )])


class PolarisRedArenaRewardFunction:
    def __init__(
            self,
            reward_scales: dict | None,
            hash_counts: HashCounts,
            inital_gamestate: GameState,
            laziness_delta_t: int,
            laziness_threshold: int,
    ):
        """
        This class takes care of computing rewards.
        :param reward_scales: scales for each goal.
        :param inital_gamestate: initial state of the game to setup the reward function.
        :param laziness_delta_t: terminates the episode if the number of reward collected in delta_t steps is below the
            set threshold.
        :param laziness_threshold: reward amount threshold before the agent is not set as lazy.
        """

        self.scales = ArenaGoals() if reward_scales is None else ArenaGoals(** reward_scales)
        self.delta_goals = ArenaGoals()

        self._cumulated_rewards_history = [None] * laziness_delta_t
        self._cumulated_rewards = ArenaGoals()
        self._previous_gamestate = inital_gamestate
        self._previous_goals = self._extract_goals(inital_gamestate)

    def _extract_goals(
            self,
            gamestate: GameState
    ) -> ArenaGoals:


        # We lose when we are no longer in battle and the map changed
        # if the map did not change, we won the battle
        win = int(not gamestate.is_in_battle) # 1 if the battle is done
        if gamestate.map != self._previous_gamestate.map: # the map changed
            win = - win

        num_fainted = sum([
            int(hp == 0.) for hp in gamestate.party_hp
        ])


        self._previous_gamestate = gamestate
        return ArenaGoals(
            experience=gamestate._additional_memory.statistics.episode_party_lvl_sum_no_lead,
            fainting=num_fainted,
            win=win

        )

    def _get_goal_updates(
            self,
            goals: ArenaGoals
    ) -> ArenaGoals:
        return _get_goals_delta(self._previous_goals, goals)

    def compute_step_rewards(
            self,
            gamestate: GameState
    ) -> float:
        goals = self._extract_goals(gamestate)
        goal_updates = self._get_goal_updates(goals)

        rewards = ArenaGoals(
            experience=goal_updates.experience,
            fainting=-np.maximum(0, goal_updates.fainting),
            win=goals.win
        )

        self._cumulated_rewards = accumulate_goal_stats(rewards, self._cumulated_rewards)
        self._cumulated_rewards_history.pop(0)
        self._cumulated_rewards_history.append(_compute_step_reward(self._cumulated_rewards, self.scales))
        self._previous_goals = goals
        return _compute_step_reward(rewards, self.scales)

    def is_lazy(self) -> bool:
        """
        :return: True if the agent recently did not collect enough rewards.
        """
        return False

    def get_metrics(self) -> dict:
        return {
            "total_rewards": self._cumulated_rewards_history[-1],
        }







