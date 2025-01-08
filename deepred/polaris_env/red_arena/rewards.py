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
    Penalty incurred when we reach episode termination/timeout
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
            inital_gamestate: GameState,

    ):
        """
        This class takes care of computing rewards in red arena.
        """

        self.scales = ArenaGoals() if reward_scales is None else ArenaGoals(** reward_scales)
        self.delta_goals = ArenaGoals()

        self._cumulated_rewards = ArenaGoals()
        self._previous_gamestate = inital_gamestate
        self._previous_goals = self._extract_goals(inital_gamestate)

    def _extract_goals(
            self,
            gamestate: GameState
    ) -> ArenaGoals:
        """
        Hint: We won/lost whenever we are no longer in battle (gamestate.is_in_battle)
              We lost if we are no longer in battle AND the map changed
        """

        self._previous_gamestate = gamestate
        return ArenaGoals(
            ...
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
            ...
        )

        self._cumulated_rewards = accumulate_goal_stats(rewards, self._cumulated_rewards)
        self._previous_goals = goals
        return _compute_step_reward(rewards, self.scales)

    def is_lazy(self) -> bool:
        """
        Terminates the episode if we are lazy.
        This does not need to be changed I think.
        """
        return False

    def get_metrics(self) -> dict:
        return {}







