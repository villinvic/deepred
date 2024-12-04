import numpy as np

from deepred.polaris_env.enums import EventFlag
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.map_dimensions import MapDimensions


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
        uint8_flag_count = round(255 * gamestate.event_flag_count / len(EventFlag))
        self.visited_tiles[gamestate.map][gamestate.pos_x, gamestate.pos_y] = uint8_flag_count


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



class AdditionalMemory(AdditionalMemoryBlock):

    def __init__(self):
        self.visited_tiles = VisitedTiles()
        self.blocks = [self.visited_tiles]

    def update(
            self,
            gamestate: GameState
    ):
        for block in self.blocks:
            block.update(gamestate)

    def reset(self):
        for block in self.blocks:
            block.reset()
