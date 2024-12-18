from dataclasses import asdict
from typing import NamedTuple, Dict

import numpy as np
import tree
from attr import dataclass

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.enums import Map
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
    badges: float = 0
    experience: float = 0
    events: float = 0
    blackout: float = 0
    heal: float = 0
    early_termination: float = 0
    # computed with map-(event flags) hash
    exploration: float = 0
    visited_maps: float = 0


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

    return sum(tree.map_structure(
        lambda r, s: r * s,
    rewards, scales
    ).values())

class PolarisRedRewardFunction:
    def __init__(
            self,
            reward_scales: dict | None,
            hash_counts: HashCounts,
            inital_gamestate: GameState
    ):
        """
        This class takes care of computing rewards.
        :param reward_scales: scales for each goal.
        :param inital_gamestate: initial state of the game to setup the reward function.
        """
        self.episode_max_party_lvl = -np.inf
        self.episode_max_event_count = 0

        self.scales = Goals() if reward_scales is None else Goals(** reward_scales)
        self.delta_goals = Goals()

        h = hash_function((inital_gamestate.map, inital_gamestate.event_flag_count, inital_gamestate.pos_x, inital_gamestate.pos_y))
        self.total_exploration = 0

        self._cumulated_rewards = Goals()
        self._previous_gamestate = inital_gamestate
        self.hash_counts = hash_counts
        self.hash_counts.visit(h)
        self.visited_hashes = set()
        self.visited_event_hashes = set()
        self.visited_maps = {Map.OAKS_LAB, Map.BLUES_HOUSE, Map.REDS_HOUSE_2F, Map.REDS_HOUSE_1F, Map.PALLET_TOWN}
        
        self._previous_goals = self._extract_goals(inital_gamestate)

    def _extract_goals(
            self,
            gamestate: GameState
    ) -> Goals:
        #     seen_pokemons: float = 0
        seen_pokemons = gamestate.species_seen_count
        #     badges: float = 0
        badges = sum(gamestate.badges)
        #     experience: float = 0
        party_level = sum(gamestate.party_level)
        if party_level > self.episode_max_party_lvl:
            self.episode_max_party_lvl = party_level

        event_count = gamestate.event_flag_count
        if event_count > self.episode_max_event_count:
            self.episode_max_event_count = event_count
            
        if gamestate.map not in self.visited_maps:
            self.visited_maps.add(gamestate.map)

        if gamestate.map not in (Map.OAKS_LAB, Map.BLUES_HOUSE, Map.REDS_HOUSE_2F, Map.REDS_HOUSE_1F, Map.PALLET_TOWN):
            h_all = hash_function((gamestate.map, gamestate.event_flag_count,  gamestate.pos_x, gamestate.pos_y))
            h = hash_function((gamestate.map,  gamestate.pos_x, gamestate.pos_y))
            if h not in self.visited_hashes:
                self.total_exploration += 0.5
                self.visited_hashes.add(h)
            if h_all not in self.visited_event_hashes:
                self.total_exploration += 1.
                self.visited_event_hashes.add(h_all)

            # if (gamestate.map not in (Map.PALLET_TOWN, Map.OAKS_LAB, Map.BLUES_HOUSE, Map.REDS_HOUSE_1F, Map.REDS_HOUSE_2F)):
                #self.total_exploration += self.hash_counts[h]
                self.hash_counts.visit(h)

        heal = int(
            self._previous_gamestate.current_checkpoint != gamestate.current_checkpoint
        )

        blackout = int(
            sum(self._previous_gamestate.party_hp) / self._previous_gamestate.party_count <
            sum(gamestate.party_hp) / gamestate.party_count
            and
            self._previous_gamestate.map != gamestate.map
            and
            (gamestate.is_at_pokecenter or gamestate.map == Map.PALLET_TOWN)
        )

        self._previous_gamestate = gamestate
        return Goals(
            seen_pokemons=seen_pokemons,
            badges=badges,
            experience=self.episode_max_party_lvl,
            events=self.episode_max_event_count,
            exploration=self.total_exploration,
            blackout=blackout,
            heal=heal,
            visited_maps=len(self.visited_maps)
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
            experience=goal_updates.experience,
            events=goal_updates.events,
            exploration=goal_updates.exploration,
            blackout=-np.maximum(0, goal_updates.blackout), # blackout_update = 1 when we blackout, -1 when we respawn.
            heal = np.maximum(0, goal_updates.heal),  # when we got a new checkpoint
            visited_maps = goal_updates.visited_maps
        )

        self._cumulated_rewards = accumulate_goal_stats(rewards, self._cumulated_rewards)
        self._previous_goals = goals
        return _compute_step_reward(rewards, self.scales)

    def get_metrics(self) -> dict:
        return {
            "to_pop/visited_hash": self.hash_counts.counts,
            "rewards": self._cumulated_rewards,
            "party_level_sum": self.episode_max_party_lvl
        }







