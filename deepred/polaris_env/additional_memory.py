import numpy as np

from deepred.polaris_env.pokemon_red.enums import EventFlag, Map, ProgressionFlag
from deepred.polaris_env.pokemon_red.map_dimensions import MapDimensions
from deepred.polaris_utils.counting import hash_function


class AdditionalMemoryBlock:

    def update(
            self,
            gamestate: "GameState"
    ):
        """
        Updates the memory object
        """
        pass

    def reset(self):
        """
        Resets the memory object
        """
        self.__init__()


class VisitedTiles(AdditionalMemoryBlock):
    def __init__(self):
        self.visited_tiles = dict()
        self.coord_map_hashes = dict()
        self.coord_map_event_hashes = dict()

    def update(
            self,
            gamestate: "GameState"
    ):
        """
        Keeps track of tiles previously visited.
        """
        coord_map_event_hash = hash_function((gamestate.map, gamestate.event_flag_count,  gamestate.pos_x, gamestate.pos_y))
        coord_map_hash = hash_function((gamestate.map,  gamestate.pos_x, gamestate.pos_y))
        if coord_map_hash not in self.coord_map_hashes:
            self.coord_map_hashes[coord_map_hash] = 0
        self.coord_map_hashes[coord_map_hash] += 1

        if coord_map_event_hash not in self.coord_map_event_hashes:
            self.coord_map_event_hashes[coord_map_event_hash] = 0
        self.coord_map_event_hashes[coord_map_event_hash] += 1

        if gamestate.map not in self.visited_tiles:
            map_w, map_h = MapDimensions[gamestate.map].shape
            self.visited_tiles[gamestate.map] = np.ones((map_h, map_w), dtype=np.int32) * -1e8

        self.visited_tiles[gamestate.map][gamestate.pos_y, gamestate.pos_x] = gamestate.step

    def is_overvisited(self, gamestate: "GameState") -> bool:
        coord_map_event_hash = hash_function((gamestate.map, gamestate.event_flag_count,  gamestate.pos_x, gamestate.pos_y))
        return False #self.coord_map_event_hashes.get(coord_map_event_hash, 0) > 10000


    def get(
            self,
            gamestate: "GameState"
    ) -> np.ndarray:
        """
        """
        return self.visited_tiles[gamestate.map]



class PokecenterCheckpoints(AdditionalMemoryBlock):

    # the pokecenter ids are not the same as pokecenter map ids.
    pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]

    def __init__(self):
        self.registered_checkpoints = [0] * len(self.pokecenter_ids)

    def update(
            self,
            gamestate: "GameState"
    ):
        last_checkpoint = gamestate.current_checkpoint
        if last_checkpoint not in self.pokecenter_ids:
            return
        self.registered_checkpoints[self.pokecenter_ids.index(last_checkpoint)] = 1


class FlagHistory(AdditionalMemoryBlock):

    def __init__(
            self,
            flag_history_length: int = 16
    ):
        """
        They block some maps.
        """
        self.flag_history_length = flag_history_length
        self.flag_history = [0] * flag_history_length
        self.stepstamps = [0] * flag_history_length

        self._previous_event_flags = None


    def update(
            self,
            gamestate: "GameState"
    ):
        if self._previous_event_flags is not None:
            new_events = np.argwhere(gamestate.event_flags-self._previous_event_flags == 1)
            for new_event in new_events:
                self.flag_history.pop(0)
                self.flag_history.append(new_event[0])
                self.stepstamps.pop(0)
                self.stepstamps.append(gamestate.step)

        self._previous_event_flags = gamestate.event_flags


class MapHistory(AdditionalMemoryBlock):

    def __init__(
            self,
            map_history_length: int = 10,
    ):
        """
        Keeps track of the latest maps visited.
        # TODO: they also give the number of steps since each map was last visited.

        :param map_history_length: Length of the history
        """
        self.visited_maps = {Map.OAKS_LAB, Map.BLUES_HOUSE, Map.REDS_HOUSE_1F, Map.REDS_HOUSE_2F}
        self.map_history_length = map_history_length
        self.map_history = [Map.UNUSED_MAP_69] * map_history_length

    def update(
            self,
            gamestate: "GameState"
    ):
        if gamestate.map not in self.visited_maps:
            self.visited_maps.add(gamestate.map)
        if gamestate.map == self.map_history[-1]:
            return

        self.map_history.pop(0)
        self.map_history.append(gamestate.map)
        

class BattleStaling(AdditionalMemoryBlock):
    
    def __init__(self):
        """
        Detects if we are staling in a battle (to avoid blackouts)
        """
        self.staling_count = 0
        self.prev_turn_count = -1
    
    def update(
            self,
            gamestate: "GameState"
    ):
        if not gamestate.is_in_battle:
            self.reset()
            return
        
        curr_turn = gamestate.battle_turn
        
        if curr_turn == self.prev_turn_count:
            self.staling_count += 1
        else:
            self.staling_count = 0
            
        self.prev_turn_count = curr_turn
    
    def is_battle_staling(self) -> bool:
        return self.staling_count > 50
            

class GoodPokemonInBoxCache(AdditionalMemoryBlock):

    def __init__(
            self,
    ):
        """
        Checks when number of pokemons in boxes changes if we have a better pokemon there
        """
        self.better_pokemon_in_box = False
        self._prev_box_count = 0

    def update(
            self,
            gamestate: "GameState"
    ):
        if gamestate.box_pokemon_count == self._prev_box_count:
            return

        party_stat_sums = gamestate.party_pokemon_stat_sums
        worst_party_pokemon_index = gamestate.worst_party_pokemon_index
        box_stat_sums = gamestate.box_pokemon_stat_sums
        best_box_pokemon_index = gamestate.best_box_pokemon_index
        self.better_pokemon_in_box = party_stat_sums[worst_party_pokemon_index] < box_stat_sums[best_box_pokemon_index]


class Statistics(AdditionalMemoryBlock):
    def __init__(self):
        self.episode_max_party_lvl_sum = -np.inf
        self.episode_max_event_count = 0
        self.episode_max_opponent_level = 5

    def update(
                self,
                gamestate: "GameState"
    ):
        if self.episode_max_opponent_level < gamestate.opponent_party_max_level:
            self.episode_max_opponent_level = gamestate.opponent_party_max_level
        party_level_sum = sum(gamestate.party_level)
        if party_level_sum > self.episode_max_party_lvl_sum:
            self.episode_max_party_lvl_sum = party_level_sum
        if self.episode_max_event_count < gamestate.event_flag_count:
            self.episode_max_event_count = gamestate.event_flag_count

class AdditionalMemory(AdditionalMemoryBlock):

    def __init__(
            self,
            map_history_length: int = 10,
            flag_history_length: int = 16

    ):
        """
        Adds some other information about the game which we cannot track with the GameState object (typically stuff
        that we have to remember but is not in the game's ram.)
        :param map_history_length:
        """

        self.visited_tiles = VisitedTiles()
        self.pokecenter_checkpoints = PokecenterCheckpoints()
        self.map_history = MapHistory(map_history_length)
        self.flag_history = FlagHistory(flag_history_length)
        self.good_pokemon_in_box = GoodPokemonInBoxCache()
        self.statistics = Statistics()
        self.battle_staling_checker = BattleStaling()

        self.blocks = [self.visited_tiles, self.pokecenter_checkpoints, self.map_history, self.flag_history,
                       self.good_pokemon_in_box, self.statistics, self.battle_staling_checker]

    def update(
            self,
            gamestate: "GameState"
    ):
        for block in self.blocks:
            block.update(gamestate)

    def reset(self):
        for block in self.blocks:
            block.reset()
