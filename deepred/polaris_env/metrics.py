from collections import defaultdict
from typing import Union

import numpy as np

from deepred.polaris_env.pokemon_red.enums import Pokemon
from deepred.polaris_env.gamestate import GameState


class PolarisRedMetrics:
    def __init__(
            self
    ):
        self.metrics = defaultdict(float)

        # keeps track of how many steps we stay on each map-(event flags) pair
        self.items_used_in_battle = 0
        self.shopping = 0

        self._prev_gamestate : Union[None, GameState] = None


    def update(
            self,
            gamestate: GameState,
    ):
        if self._prev_gamestate is not None:
            total_items_delta = sum(gamestate.bag_items.values()) - sum(self._prev_gamestate.bag_items.values())
            if total_items_delta != 0:
                if gamestate.is_in_battle:
                    self.items_used_in_battle += 1
                elif gamestate.is_at_pokemart:
                    self.shopping += 1


        self._prev_gamestate = gamestate

    def get_metrics(
            self,
            final_gamestate: GameState
    ):
        pokemons = defaultdict(int)

        for pokemon in Pokemon:
            if pokemon not in (Pokemon.M_GLITCH, Pokemon.NONE) and pokemon in final_gamestate.party_pokemons:
                pokemons[pokemon.name] += 100
            else:
                pokemons[pokemon.name] = 0

        return {
            "shopping": self.shopping,
            "num_visited_maps": len(final_gamestate._additional_memory.map_history.visited_maps),
            "items_used_in_battle": self.items_used_in_battle,
            "party_pokemons": np.array(pokemons), # pass a numpy array to interpret this as a barplot
            "party_count": final_gamestate.party_count,  # pass a numpy array to interpret this as a barplot
            "party_level_min": np.min(final_gamestate.party_level),  # pass a numpy array to interpret this as a barplot

            "money": final_gamestate.player_money,
            "num_triggered_flags": np.sum(final_gamestate.event_flags)
        }




