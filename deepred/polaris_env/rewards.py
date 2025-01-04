from dataclasses import asdict
from typing import NamedTuple, Dict

import numpy as np
import tree
from attr import dataclass

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.enums import Map, FixedPokemonType
from deepred.polaris_utils.counting import HashCounts, hash_function


class Goals(NamedTuple):
    """
    Rewards collected by the agent each step.
    The remaining rewards will be on the Polaris side
    (we need to communicate visitation stats between the learner and the workers).

    # Rewards used by:
    # https://github.com/CJBoey/PokemonRedExperiments1/blob/master/baselines/boey_baselines2/red_gym_env.py
    # secret switch states
    # first time healing in a new pokecenter
    # getting hms (bonus for usable ones)
    # healing / blackouts
    # triggering events
    # special key items
    # some "special" rewards

    """
    seen_pokemons: float = 0
    party_building: float = 0
    badges: float = 0
    experience: float = 0
    events: float = 0
    blackout: float = 0
    fainting: float = 0
    heal: float = 0
    early_termination: float = 0
    opponent_level: float = 0
    # computed with map-(event flags) hash
    exploration: float = 0
    shopping: float = 0
    box_usage: float = 0

    battle_staling: float = 0


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

        self.scales = Goals() if reward_scales is None else Goals(** reward_scales)
        self.delta_goals = Goals()

        self.total_exploration = 0
        self.hash_counts = hash_counts
        self.laziness_threshold = laziness_threshold
        self._cumulated_rewards_history = [None] * laziness_delta_t

        self._cumulated_rewards = Goals()
        self._previous_gamestate = inital_gamestate
        self._previous_goals = self._extract_goals(inital_gamestate)

    def _extract_goals(
            self,
            gamestate: GameState
    ) -> Goals:

        overvisited = int(gamestate._additional_memory.visited_tiles.is_overvisited(gamestate))
        self.total_exploration = 0.5*(len(gamestate._additional_memory.visited_tiles.coord_map_hashes)
                                       + len(gamestate._additional_memory.visited_tiles.coord_map_event_hashes)) - overvisited

        blackout = int(
            sum(self._previous_gamestate.party_hp) / self._previous_gamestate.party_count <
            sum(gamestate.party_hp) / gamestate.party_count
            and
            self._previous_gamestate.map != gamestate.map
            and
            (gamestate.is_at_pokecenter or gamestate.map == Map.PALLET_TOWN)
        )

        num_fainted = sum([
            int(hp == 0.) for hp in gamestate.party_hp
        ])

        # kinda hard to optimise this I guess
        # TODO: when swapping pc mons, care about types and evolutions.
        type_diversity = len({ptype for ptype in gamestate.party_types.flatten() if ptype != FixedPokemonType.NO_TYPE})

        self._previous_gamestate = gamestate
        return Goals(
            seen_pokemons=gamestate.species_seen_count,
            party_building=type_diversity,
            badges=gamestate.badge_count,
            experience=gamestate._additional_memory.statistics.episode_party_lvl_sum_no_lead,
            events=gamestate._additional_memory.statistics.episode_max_event_count,
            exploration=self.total_exploration,
            blackout=blackout,
            fainting=num_fainted,
            opponent_level=gamestate._additional_memory.statistics.episode_max_opponent_level,
            heal=gamestate.visited_pokemon_centers_count,
            shopping=sum(gamestate.bag_items.values()),
            box_usage=int(gamestate.has_better_pokemon_in_box),
            battle_staling=int(gamestate._additional_memory.battle_staling_checker.is_battle_staling())
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
            party_building=goal_updates.party_building,
            badges=goal_updates.badges,
            experience=np.clip(goal_updates.experience, 0, 1),
            events=goal_updates.events,
            exploration=goal_updates.exploration,
            blackout=-np.maximum(0, goal_updates.blackout), # blackout_update = 1 when we blackout, -1 when we respawn.
            fainting=-np.maximum(0, goal_updates.fainting),
            opponent_level=goal_updates.opponent_level,
            heal = goal_updates.heal,  # when we got a new checkpoint
            shopping = int(goal_updates.shopping != 0 and gamestate.is_at_pokemart),
            box_usage= abs(goal_updates.box_usage), # +1 when catching good pokemon and +1 for retrieving it in the box.
            battle_staling = -goals.battle_staling
        )

        self._cumulated_rewards = accumulate_goal_stats(rewards, self._cumulated_rewards)
        self._cumulated_rewards_history.pop(0)
        self._cumulated_rewards_history.append(_compute_step_reward(self._cumulated_rewards, self.scales))
        self._previous_goals = goals
        return _compute_step_reward(rewards, self.scales)

    def laziness(self) -> float:
        """
        :return: 1 if lazy, close to 0 if not lazy.
        """
        if self._cumulated_rewards_history[0] is None:
            return 0
        dr = self._cumulated_rewards_history[-1] - self._cumulated_rewards_history[0]
        gap = (dr - self.laziness_threshold) / self.laziness_threshold
        gap = np.clip(gap, 0, 50)
        laziness = 1 - gap / 50
        return laziness

    def is_lazy(self) -> bool:
        """
        :return: True if the agent recently did not collect enough rewards.
        """
        if self._cumulated_rewards_history[0] is None:
            return False
        return self._cumulated_rewards_history[-1] - self._cumulated_rewards_history[0] < self.laziness_threshold

    def get_metrics(self) -> dict:
        return {
            "to_pop/visited_hash": self.hash_counts.counts,
            "rewards": self._cumulated_rewards,
            "badges_collected_from_checkpoint": self._cumulated_rewards.badges,
            "total_rewards": self._cumulated_rewards_history[-1],
        }







