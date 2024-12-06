import numpy as np

from deepred.polaris_env.pokemon_red.enums import EventFlag, Map
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.map_dimensions import MapDimensions


class AdditionalMemoryBlock:

    def update(
            self,
            gamestate: GameState
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

    def update(
            self,
            gamestate: GameState
    ):
        """
        Keeps track of tiles previously visited.
        """
        if gamestate.map not in self.visited_tiles:
            self.visited_tiles[gamestate.map] = np.zeros(MapDimensions[gamestate.map].shape, dtype=np.uint8)

        # TODO: unsure how to set the values.
        #   We need something that let us know we already walked in some places at some point in time,
        #   The agent will have to go multiple times to some places.
        #   Could use step counter.
        uint8_flag_count = round(255 * gamestate.event_flag_count / len(EventFlag))
        self.visited_tiles[gamestate.map][gamestate.pos_x, gamestate.pos_y] = uint8_flag_count

    def get(
            self,
            gamestate: GameState
    ):
        """
        recent visited tiles are set to 255, decrementing to 0 for each event that was triggered since the visit to the
        tile.
        """
        uint8_flag_count = round(255 * gamestate.event_flag_count / len(EventFlag))
        return 255 - (uint8_flag_count - self.visited_tiles[gamestate.map])



class PokecenterCheckpoints(AdditionalMemoryBlock):

    # the pokecenter ids are not the same as pokecenter map ids.
    pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]

    def __init__(self):
        self.registered_checkpoints = [0] * len(self.pokecenter_ids)

    def update(
            self,
            gamestate: GameState
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
            gamestate: GameState
    ):
        if self._previous_event_flags is not None:
            new_events = np.argwhere(gamestate.event_flags-self._previous_event_flags == 1)
            for new_event in new_events:
                self.flag_history.pop(0)
                self.flag_history.append(new_event)
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
        self.map_history_length = map_history_length
        self.map_history = [Map.UNUSED_MAP_69] * map_history_length

    def update(
            self,
            gamestate: GameState
    ):
        if gamestate.map in self.map_history:
            return

        self.map_history.pop(0)
        self.map_history.append(gamestate.map)


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
            gamestate: GameState
    ):
        if gamestate.box_pokemon_count == self._prev_box_count:
            return

        party_stat_sums = gamestate.party_pokemon_stat_sums
        worst_party_pokemon_index = gamestate.worst_party_pokemon_index
        box_stat_sums = gamestate.box_pokemon_stat_sums
        best_box_pokemon_index = gamestate.best_box_pokemon_index
        self.better_pokemon_in_box = party_stat_sums[worst_party_pokemon_index] < box_stat_sums[best_box_pokemon_index]


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

        self.blocks = [self.visited_tiles, self.pokecenter_checkpoints, self.map_history, self.flag_history,
                       self.good_pokemon_in_box]

    def update(
            self,
            gamestate: GameState
    ):
        for block in self.blocks:
            block.update(gamestate)

    def reset(self):
        for block in self.blocks:
            block.reset()
