from collections import defaultdict
from typing import Union

import numpy as np

from deepred.polaris_env.pokemon_red.enums import Pokemon
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_utils.counting import hash_function


class PolarisRedMetrics:
    def __init__(
            self
    ):

        self.metrics = defaultdict(float)

        # keeps track of how many steps we stay on each map-(event flags) pair
        self.visited_hash = set()
        self.visitated_maps = set()
        self.items_bought_in_mart = 0
        self.items_used_in_battle = 0
        self.items_tossed = 0 # is there cases where the player has to give an item ?

        self._prev_gamestate : Union[None, GameState] = None


    def update(
            self,
            gamestate: GameState,
    ):
        if self._prev_gamestate is not None:

            map_event_flag_hash = hash_function((gamestate.map, gamestate.event_flags))
            self.visited_hash.add(map_event_flag_hash)
            self.visitated_maps.add(gamestate.map)
            total_items_delta = sum(gamestate.bag_items.values()) - sum(self._prev_gamestate.bag_items.values())
            if total_items_delta != 0 and gamestate.is_in_battle or gamestate.is_at_pokemart():
                if total_items_delta > 0:
                    self.items_bought_in_mart += total_items_delta
                elif total_items_delta < 0:
                    self.items_used_in_battle += total_items_delta
            elif total_items_delta < 0:
                self.items_tossed += total_items_delta


        self._prev_gamestate = gamestate

    def get_metrics(
            self,
            final_gamestate: GameState
    ):
        pokemons = defaultdict(int)
        for pokemon in final_gamestate.party_pokemons:
            if pokemon not in (Pokemon.M_GLITCH, Pokemon.NONE):
                pokemons[pokemon.name] += 1

        return {
            "to_pop/visited_hash": self.visited_hash,
            "num_visited_hash": len(self.visited_hash),
            "num_visited_maps": len(self.visitated_maps),
            #"visited_maps": self.visitated_maps, # sets are not supported as metrics.
            "items_used_in_battle": self.items_used_in_battle,
            "items_bought_in_mart": self.items_bought_in_mart,
            "party_pokemons": pokemons,
            "money": final_gamestate.player_money,
            "num_triggered_flags": np.sum(final_gamestate.event_flags)
        }




