from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict, Tuple

import numpy as np
from PIL import Image

from deepred.polaris_env.additional_memory import AdditionalMemory
from deepred.polaris_env.pokemon_red.enums import StartMenuItem, Map, RamLocation, Pokemon, BagItem, EventFlag, TileSet, \
    DataStructDimension, Badges, BattleType, ProgressionFlag, Orientation, Move
from deepred.polaris_env.pokemon_red.map_dimensions import MapDimensions
from deepred.polaris_env.pokemon_red.map_warps import MapWarps, MapWarp, NamedWarpIds
from deepred.polaris_env.pokemon_red.mart_info import MartInfos, MartInfo
from deepred.polaris_env.pokemon_red.move_info import MovesInfo
from deepred.polaris_env.pokemon_red.pokemon_stats import PokemonStats, PokemonBaseStats
from deepred.polaris_env.utils import cproperty

B = 256
B2 = B**2

EVENT_ADDRESS_ARRAY = np.array([
    e.value for e in ProgressionFlag
])

stats_level_scale = 50

class GameState:
    

    def __repr__(self):
        class_name = self.__class__.__name__
        properties = []
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, cproperty):  # Check if the attribute is a cproperty
                value = getattr(self, name, None)  # Get the cproperty's value
                properties.append(f"{name}={value!r}")
        return f"{class_name}({', '.join(properties)})"

    def dump(
            self,
            dump_path: Path,
            msg: str = "",
    ):
        """
        Serializes the GameState into two files:
        - A png image for the screen.
        - A log file for the gamestate properties + custom msg.

        :param file_prefix: Prefix for the output file names.
        """
        # Save the screen as an image file
        image_file = dump_path.with_suffix(".png")
        Image.fromarray(self.screen).save(image_file)

        # Collect the properties of the object
        log_file = dump_path.with_suffix(".log")
        with open(log_file, "w") as f:
            f.write(f"dump message: '{msg}'.\n")
            for name in dir(self.__class__):
                attr = getattr(self.__class__, name, None)
                if isinstance(attr, cproperty):
                    value = getattr(self, name, None)
                    f.write(f"{name}: {value}\n")

    def __init__(
            self,
            console: "GBConsole"
    ):
        """
        Utility class for grabbing ram data.
        For now, this class is "lazy", as it does not load data from the ram as long as it is not needed.
        :param console: the console from which the game ram will be read.
        """
        self._ram = console.memory
        self._ram_helper = console._ram_helper
        self._frame = console._frame
        self.step = console._step
        self._additional_memory: AdditionalMemory = console._additional_memory
        self.screen = np.uint8(console.screen.ndarray[:, :, 0])


        # we need to fetch those states no matter what. We need them for skipping frames
        self.start_menu_item
        self.is_in_battle

    def is_skippable_frame(self) -> bool:
        """
        :return: Whether this frame can be skipped.
        """
        return self.is_skippable_textbox()

    def is_skippable_textbox(self) -> bool:
        """
        :return: True if we can skip the textbox by ticking
        """
        # TODO: find reliable way to skip dialogs...
        return self.is_skippable_battle_frame() or self.is_dialog_frame()


    def is_skippable_battle_frame(self):
        # We skip the frame if we are in battle, have a certain box id open or if we are in the party menu.
        return (
                self.is_in_battle
                and not (
                self.textbox_id in {11, 12, 13, 20}
                or
                self.is_playing_party_animations > 0
                )
        )

    def is_skippable_party_menu(self) -> bool:
        """
        :return: True if we are looking at the party screen
        """
        # TODO: update this
        return self.is_playing_party_animations > 0

    @cproperty
    def skippable_with_a_press(self):
        """
        Normally, with modded rom, should not need this.
        """
        return (
                self._read(0xc4f2) == 238
                and
                self._read(RamLocation.MENU_STATE) == 0
                and
                self.textbox_id == 1
                and
                self.hdst_map != 0
        )

    @cproperty
    def is_in_battle(self) -> bool:
        """
        :return: True if in battle
        """
        # Battle type is 0 when not in battle.
        return self.battle_type != BattleType.NONE

    def is_dialog_frame(self) -> bool:
        """
        :return: True if we are currently at a dialog frame. Utility for skipping dialogs.
        """
        return (
                self.start_menu_item == StartMenuItem.UNSELECTED
                and not self.is_in_battle
                and self.open_text_box
                and (not self.textbox_id in {6, 11, 12, 13, 14, 20, 28})
                and not self.is_playing_party_animations == 64
                and not self.open_town_map
                and (self.item_cursor_x, self.item_cursor_y) == (0, 0)
        )

    def is_transition_screen(self) -> bool:
        """
        :return: True if we are in a transition screen (black, white, gray...)
        """
        # This uses pixels, we want to avoid that when using headless consoles
        # (they render the pixels once per agent action)
        min_color = np.min(self.screen)
        max_color = np.max(self.screen)

        return (max_color - min_color) < 20

    @cproperty
    def is_at_pokecenter(self) -> bool:
        """
        :return: True if we are at a pokecenter.
        """
        return self.map in [
            Map.MT_MOON_POKECENTER, Map.PEWTER_POKECENTER,
            Map.CERULEAN_POKECENTER, Map.VIRIDIAN_POKECENTER, Map.CELADON_POKECENTER,
            Map.ROCK_TUNNEL_POKECENTER, Map.LAVENDER_POKECENTER, Map.FUCHSIA_POKECENTER,
            Map.CINNABAR_POKECENTER, Map.SAFFRON_POKECENTER, Map.VIRIDIAN_POKECENTER,
            Map.INDIGO_PLATEAU_LOBBY
        ]

    @cproperty
    def is_pc_room(self) -> bool:
        """
        Is this a room where we have a pc access?
        """
        return self.is_at_pokecenter or self.map in [
            Map.CELADON_MANSION_2F, Map.SILPH_CO_11F, Map.CINNABAR_LAB_FOSSIL_ROOM
        ]

    @cproperty
    def can_use_pc(self) -> bool:
        """
        Can we use the pc now (right orientation, location and if we have pokemons in the pc)
        TODO: mabye allow to dump pokemons even if no pokemon in box.
        """
        return (
                self.player_orientation == Orientation.UP
                and
                (self.pos_x, self.pos_y) in pc_locations.get(self.map, pokecenter_pc_location)
        )

    @cproperty
    def is_at_pokemart(self) -> bool:
        """
        :return: True if we are at a pokemart.
        """
        return self.map in [
            Map.PEWTER_MART, Map.FUCHSIA_MART, Map.SAFFRON_MART,
            Map.CERULEAN_MART, Map.VIRIDIAN_MART, Map.VERMILION_MART,
            Map.CINNABAR_MART, Map.LAVENDER_MART, Map.CINNABAR_MART_COPY
        ]

    def _read(
            self,
            addr: Union[RamLocation, slice, int]
    ) -> Union[int, List]:
        """
        Reads the game's RAM at the given address.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr.
        """
        if isinstance(addr, RamLocation):
            addr = addr.value

        return self._ram[addr]

    def _read_double(
            self,
            addr: Union[RamLocation, int, slice]
    ) -> Union[int, List]:
        """
        Reads the game's RAM at the given address as a double.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr as a double.
        """
        if isinstance(addr, (RamLocation, int)):
            return 256 * self._read(addr) + self._read(addr + 1)

        s1 = slice(addr.start, addr.stop, 2)
        s2 = slice(addr.start+1, addr.stop+1, 2)

        return B * self._read(s1) + self._read(s2)

    def _read_triple(
            self,
            addr: Union[RamLocation, int, slice]
    ) -> Union[int, List]:
        """
        Reads the game's RAM at the given address as a triple.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr as a triple.
        """
        if isinstance(addr, int):
            return B2 * self._read(addr) + B * self._read(addr + 1) + self._read(addr + 2)

        s1 = slice(addr.start, addr.stop, 3)
        s2 = slice(addr.start + 1, addr.stop + 1, 3)
        s3 = slice(addr.start + 2, addr.stop + 2, 3)

        return B2 * self._read(s1) + B * self._read(s2) + self._read(s3)

    def _read_bcd(
            self,
            addr: Union[RamLocation, int, slice]
    ) -> int:
        """
        Reads the game's RAM at the given address as a binary coded decimal.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr as a bcd.
        """
        if isinstance(addr, int):
            n = self._read(addr)
            return 10 * ((n >> 4) & 0x0f) + (n & 0x0f)
        raise NotImplementedError("bcd read for slices is not implemented")

    def _read_bitcount(
            self,
            addr: Union[RamLocation, int, slice]
    ) -> int:
        """
        Reads the game's RAM at the given address as a binary coded decimal.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr as a bcd.
        """
        if isinstance(addr, int):
            return self._read(addr).bit_count()
        raise NotImplementedError("bitcount read for slices is not implemented")

    @cproperty
    def frame(self) -> int:
        return self._frame

    @cproperty
    def current_checkpoint(self) -> int:
        """
        Returns the respawn point if we blackout.
        """
        return self._read(RamLocation.POKECENTER_CHECKPOINT)

    @cproperty
    def instant_text(self) -> bool:
        """
        :return: True if we are using instant text.
        """
        return self._read(RamLocation.INSTANT_TEXT) & (2**6) > 0

    @cproperty
    def open_menu(self) -> int:
        """
        :returns: True if a menu is open.
        """
        return self._read(RamLocation.MENU_STATE) == 0

    @cproperty
    def walk_bike_surf_state(self) -> int:
        """
        :return: state of the player when walking, surfing, biking
        """
        return self._read(RamLocation.WALKBIKESURF_STATE)

    @cproperty
    def is_playing_party_animations(self) -> int:
        """
        When opening the party screen, animations for pokemons play, this reads the index of the sprite.
        :return: Returns the index of the sprite shown, 0 if not played
        """
        return self._read(RamLocation.PARTY_MENU_ANIMATION_INDEX)

    @cproperty
    def battle_type(self) -> BattleType:
        """
        :return: type of battle, not in battle, wild, trainer.
        """
        # Battle type is 0 when not in battle.
        return BattleType(self._read(RamLocation.BATTLE_TYPE))

    @cproperty
    def battle_kind(self) -> int:
        """
        :return: kind of battle, not in battle, gym leader, safari, etc.
        """
        # Battle type is 0 when not in battle.
        return BattleType(self._read(RamLocation.BATTLE_KIND))

    @cproperty
    def player_money(self) -> int:
        """
        The player current money
        """
        return (100 ** 2 * self._read_bcd(RamLocation.MONEY3) +
                100 * self._read_bcd(RamLocation.MONEY2) +
                self._read_bcd(RamLocation.MONEY1))

    @cproperty
    def map(self) -> Map:
        """
        The current map id.
        """
        return Map(self._read(RamLocation.MAP_ID))

    @cproperty
    def open_town_map(self) -> bool:
        """
        Returns true if we are looking at the town map now.
        """
        return self._read(RamLocation.TOWN_MAP_STATE) == 205

    @cproperty
    def pokemons(self) -> np.ndarray:
        """
        Party and opponent pokemons.
        """
        species = np.zeros((12,), dtype=np.uint8)

        for i in range(self.party_count):
            species[i] = self._read(RamLocation.PARTY_0_SPECIES + i * DataStructDimension.POKEMON_STATS)

        if self.battle_type == BattleType.WILD:
            species[6] = self._read(RamLocation.WILD_POKEMON_SPECIES)

        elif self.battle_type == BattleType.TRAINER:
            for i in range(self.opponent_party_count):
                species[i+6] = self._read(RamLocation.OPPONENT_POKEMON_0_SPECIES + i * DataStructDimension.POKEMON_STATS)

        return species

    @cproperty
    def party_pokemons(self) -> List[Pokemon]:
        """
        Party Pokemons as enums.
        """
        return [Pokemon(pid) for pid in self.pokemons[:6]]

    @cproperty
    def party_count(self) -> int:
        """
        :return: The number of pokemons in the current party
        """
        return self._read(RamLocation.PARTY_COUNT)

    @cproperty
    def opponent_party_count(self) -> int:
        """
        :return: The number of pokemons in the opponent's party
        """
        return self._read(RamLocation.OPPONENT_PARTY_COUNT)

    @cproperty
    def pokemon_types(self) -> np.ndarray:
        """
        Array of shape (12, 2)
        Types for party and opponent pokemons.
        :return:
        """
        types = np.zeros((12, 2), dtype=np.uint8)

        for i in range(self.party_count):
            type1 = self._read(RamLocation.PARTY_0_TYPE0 + i * DataStructDimension.POKEMON_STATS)
            type2 = self._read(RamLocation.PARTY_0_TYPE1 + i * DataStructDimension.POKEMON_STATS)
            types[i] = type1, type2

        if self.battle_type == BattleType.WILD:

            type1 = self._read(RamLocation.WILD_POKEMON_TYPE0)
            type2 = self._read(RamLocation.WILD_POKEMON_TYPE1)
            types[6] = type1, type2

        elif self.battle_type == BattleType.TRAINER:
            for i in range(self.opponent_party_count):
                type1 = self._read(RamLocation.OPPONENT_POKEMON_0_TYPE0 + i * DataStructDimension.POKEMON_STATS)
                type2 = self._read(RamLocation.OPPONENT_POKEMON_0_TYPE1 + i * DataStructDimension.POKEMON_STATS)
                types[i+6] = type1, type2

        return fix_pokemon_types(types)

    @cproperty
    def pokemon_moves(self) -> np.ndarray:
        """
        Array of shape (12, 4)
        Moves for party and opponent pokemons.
        """
        moves = np.zeros((12, 4), dtype=np.uint8)

        for i in range(self.party_count):
            move0 = self._read(RamLocation.PARTY_0_MOVE0 + i * DataStructDimension.POKEMON_STATS)
            move1 = self._read(RamLocation.PARTY_0_MOVE1 + i * DataStructDimension.POKEMON_STATS)
            move2 = self._read(RamLocation.PARTY_0_MOVE2 + i * DataStructDimension.POKEMON_STATS)
            move3 = self._read(RamLocation.PARTY_0_MOVE3 + i * DataStructDimension.POKEMON_STATS)

            moves[i] = move0, move1, move2, move3

        if self.battle_type == BattleType.WILD:
            move0 = self._read(RamLocation.WILD_POKEMON_MOVE0)
            move1 = self._read(RamLocation.WILD_POKEMON_MOVE1)
            move2 = self._read(RamLocation.WILD_POKEMON_MOVE2)
            move3 = self._read(RamLocation.WILD_POKEMON_MOVE3)
            moves[6] = move0, move1, move2, move3

        elif self.battle_type == BattleType.TRAINER:
            for i in range(self.opponent_party_count):
                move0 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE0 + i * DataStructDimension.POKEMON_STATS)
                move1 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE1 + i * DataStructDimension.POKEMON_STATS)
                move2 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE2 + i * DataStructDimension.POKEMON_STATS)
                move3 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE3 + i * DataStructDimension.POKEMON_STATS)
                moves[i+6] = move0, move1, move2, move3

        return moves

    @cproperty
    def pokemon_pps(self) -> np.ndarray:
        """
        Array of shape (12, 4)
        Move PPs for party and opponent pokemons.
        """
        moves = np.zeros((12, 4), dtype=np.uint8)

        for i in range(self.party_count):
            move0 = self._read(RamLocation.PARTY_0_MOVE0 + i * DataStructDimension.POKEMON_STATS)
            move1 = self._read(RamLocation.PARTY_0_MOVE1 + i * DataStructDimension.POKEMON_STATS)
            move2 = self._read(RamLocation.PARTY_0_MOVE2 + i * DataStructDimension.POKEMON_STATS)
            move3 = self._read(RamLocation.PARTY_0_MOVE3 + i * DataStructDimension.POKEMON_STATS)
            moves[i] = move0, move1, move2, move3

        if self.battle_type == BattleType.WILD:
            move0 = self._read(RamLocation.WILD_POKEMON_MOVE0_PP)
            move1 = self._read(RamLocation.WILD_POKEMON_MOVE1_PP)
            move2 = self._read(RamLocation.WILD_POKEMON_MOVE2_PP)
            move3 = self._read(RamLocation.WILD_POKEMON_MOVE3_PP)
            moves[6, :] = move0, move1, move2, move3

        elif self.battle_type == BattleType.TRAINER:
            for i in range(self.opponent_party_count):
                move0 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE0_PP + i * DataStructDimension.POKEMON_STATS)
                move1 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE1_PP + i * DataStructDimension.POKEMON_STATS)
                move2 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE2_PP + i * DataStructDimension.POKEMON_STATS)
                move3 = self._read(RamLocation.OPPONENT_POKEMON_0_MOVE3_PP + i * DataStructDimension.POKEMON_STATS)
                moves[i + 6] = move0, move1, move2, move3

        return moves

    @cproperty
    def is_swapping(self) -> bool:
        """
        :return: Are we in the swapping menu ?
        """
        return self._read(RamLocation.BATTLE_MENU_STATE) == 0x04


    @cproperty
    def swapping_position(self):
        """
        Position of the pokemon with which we are going to swap.
        :return:
        """
        if self.is_swapping:
            chosen_mon = self._read(RamLocation.POKEMON_SWAPPING_POS)
            return chosen_mon - 1

    def party_pokemon_stats_at_index(
            self,
            start_address: int,
            party_pokemon: bool,
            position: int,
            wild_pokemon: bool = False # TODO: never used ?
    ) -> List:
        """
        Returns a list of pokemon attributes (15) for party pokemons, or opponent pokemons that are not battling.
        TODO: think if we move this kind of function into the observation space module...
        TODO: Add one hot for pokemon position ? Thats what they did but not necessary.
        :param start_address: starting address of the concerned pokemon.
        :param party_pokemon: is it a pokemon from our party ?
        :param position: position of the pokemon in the team.
        :param wild_pokemon: is it a wild pokemon ?
        """
        hp = self._read_double(start_address + 1) / 250
        max_hp =self._read_double(start_address + 34) / 250
        return [
            hp,  # current hp
            int(hp > 0),  # knocked out ?
            hp / (max_hp+1e-8),
            max_hp,
            self._read(start_address + 33) / 100, # level
            self._read_double(start_address + 36) / 134, # attack
            self._read_double(start_address + 38) / 180,  # defense
            self._read_double(start_address + 40) / 140,  # speed
            self._read_double(start_address + 42) / 154,  # special
            int(wild_pokemon or (
                (party_pokemon and position == self._read(RamLocation.PARTY_SENT_OUT))
                or
                position == self._read(RamLocation.OPPONENT_POKEMON_SENT_OUT)  # should always be False ?
            )), # is sent out ?
            int(party_pokemon and position == self.swapping_position)  # are we about to swap with this pokemon ?
        ] + pokemon_status(self._read(start_address + 4))

    def opponent_sent_out_pokemon_stats_at_index(
            self,
            start_address: int,
            party_pokemon: bool,
            position: int,
            wild_pokemon: bool = False
    ) -> List:
        """
        Returns a list of pokemon attributes (15) for battling opponent pokemons
        # TODO: think if we move this kind of function into the observation space module...
        :param start_address: starting address of the concerned pokemon.
        :param party_pokemon: is it a pokemon from our party ?
        :param position: position of the pokemon in the team.
        :param wild_pokemon: is it a wild pokemon ?
        """
        hp = self._read_double(start_address + 1) / 250
        max_hp = self._read_double(start_address + 15) / 250 # max hp
        return [
            hp,  # current hp
            int(hp > 0),  # knocked out ?
            hp / max_hp,
            max_hp,  # max hp
            self._read(start_address + 14) / 100, # levels
            self._read_double(start_address + 17) / 134, # attack
            self._read_double(start_address + 19) / 180,  # defense
            self._read_double(start_address + 21) / 140,  # speed
            self._read_double(start_address + 23) / 154,  # special
            1, # sent out # could skip thoses, but they are used so that we can put together the stats of the 12 pokemons.
            0 # TODO: does this make sense ?
        ] + pokemon_status(self._read(start_address + 4))

    @cproperty
    def pokemon_attributes(self) -> np.ndarray:
        """
        An array of shape (12, 15) holding attributes for all pokemons (party and opponent).
        I feel like having every info about the opponent team is kind of cheating, but again, you can also cheat
        by checking the game online I guess.
        """
        attributes = np.zeros((12, 18), np.float32)

        for i in range(self.party_count):
            attributes[i] = self.party_pokemon_stats_at_index(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS,
                                                   party_pokemon=True, position=i, wild_pokemon=False)

        if self.battle_type == BattleType.WILD:
            attributes[6] = self.opponent_sent_out_pokemon_stats_at_index(
                        RamLocation.WILD_POKEMON_SPECIES,  # this is also the address for opponent sent out pokemons.
                        party_pokemon=False,
                        position=0,
                        wild_pokemon=True,
                    )

        elif self.battle_type == BattleType.TRAINER:
            for i in range(self.opponent_party_count):
                if i == self._read(RamLocation.OPPONENT_POKEMON_SENT_OUT):
                    attributes[i + 6] = self.opponent_sent_out_pokemon_stats_at_index(
                        RamLocation.WILD_POKEMON_SPECIES,  # this is also the address for opponent sent out pokemons.
                        party_pokemon=False,
                        position=i,
                        wild_pokemon=False,
                    )
                else:
                    attributes[i + 6] = self.party_pokemon_stats_at_index(
                        RamLocation.OPPONENT_POKEMON_0_SPECIES,  # this is also the address for opponent sent out pokemons.
                        party_pokemon=False,
                        position=i,
                        wild_pokemon=False,
                    )

        return attributes

    @cproperty
    def textbox_id(self) -> int:
        """
        The id of the current textbox.
        """
        return self._read(RamLocation.TEXTBOX_ID)

    @cproperty
    def open_text_box(self) -> bool:
        """
        :return: True if a textbox is currently open
        """
        return self._read(RamLocation.OPEN_TEXT_BOX) == 1

    @cproperty
    def box_pokemon_count(self) -> int:
        """
        The number of pokemons in the box.
        """
        return self._read(RamLocation.BOX_POKEMON_COUNT)

    @cproperty
    def item_cursor_x(self) -> int:
        """
        :return: Returns the x coordinates (on the screen) of the topmost item cursor
        """
        return self._read(RamLocation.ITEM_CURSOR_X)

    @cproperty
    def item_cursor_y(self) -> int:
        """
        :return: Returns the y coordinates (on the screen) of the topmost item cursor
        """
        return self._read(RamLocation.ITEM_CURSOR_Y)

    @cproperty
    def open_start_menu(self) -> bool:
        """
        :return: True if we are navigating in the start menu
        """
        return self._read(RamLocation.OPEN_START_MENU) == 1

    @cproperty
    def menu_item_id(self) -> int:
        """
        Current id of the highlighted menu item
        """
        return self._read(RamLocation.MENU_ITEM_ID)

    @cproperty
    def start_menu_item(self) -> StartMenuItem:
        """
        :return: Start item currently selected, returns unselected if the menu is not open.
        """

        if self.open_start_menu:
            return StartMenuItem(self.menu_item_id)

        return StartMenuItem.UNSELECTED

    # @cproperty
    # def party_hp(self) -> List[float]:
    #     """
    #     The current party pokemon health fractions (0 is fainted, 1 is full hp).
    #     """
    #     return [ self._read_double(curr_hp_addr) / (self._read_double(max_hp_addr) + 1e-8)
    #     # Addresses are not adjacent.
    #     for curr_hp_addr, max_hp_addr in [
    #                  (RamLocation.PARTY_0_HP, RamLocation.PARTY_0_MAXHP),
    #                  (RamLocation.PARTY_1_HP, RamLocation.PARTY_1_MAXHP),
    #                  (RamLocation.PARTY_2_HP, RamLocation.PARTY_2_MAXHP),
    #                  (RamLocation.PARTY_3_HP, RamLocation.PARTY_3_MAXHP),
    #                  (RamLocation.PARTY_4_HP, RamLocation.PARTY_4_MAXHP),
    #                  (RamLocation.PARTY_5_HP, RamLocation.PARTY_5_MAXHP),
    #              ]
    #     ]

    @cproperty
    def party_experience(self) -> List[float]:
        """
        The current party pokemon experiences
        """
        return [
            self._read_triple(addr)
            for addr in [RamLocation.PARTY_0_EXP, RamLocation.PARTY_1_EXP, RamLocation.PARTY_2_EXP,
                         RamLocation.PARTY_3_EXP, RamLocation.PARTY_4_EXP, RamLocation.PARTY_5_EXP]
        ]

    @cproperty
    def party_level(self) -> List[int]:
        """
        The current party pokemon levels
        """

        return [
            self._read(addr)
            for addr in [RamLocation.PARTY_0_LEVEL, RamLocation.PARTY_1_LEVEL, RamLocation.PARTY_2_LEVEL,
                         RamLocation.PARTY_3_LEVEL, RamLocation.PARTY_4_LEVEL, RamLocation.PARTY_5_LEVEL]
        ]

    @cproperty
    def sent_out(self) -> int:
        """
        The pokemon currently sent out in battle
        'wBattleMonPartyPos' is not used for some reason.
        We use 'wPlayerMonNumber' which appear to have the same function.
        """
        return self._read(RamLocation.SENT_OUT_PARTY_POS)

    @cproperty
    def species_caught_count(self) -> int:
        """
        The total count of pokemon species caught.
        """
        return sum([
            self._read_bitcount(addr) for addr in range(RamLocation.CAUGHT_SPECIES_START,
                                                           RamLocation.CAUGHT_SPECIES_END)
        ])

    @cproperty
    def species_seen_count(self) -> int:
        """
        The total count of pokemon species seen.
        """
        return sum([
            self._read_bitcount(addr) for addr in range(RamLocation.SEEN_SPECIES_START,
                                                           RamLocation.SEEN_SPECIES_END)
        ])

    @cproperty
    def badges(self) -> List[int]:
        """
        The total count of pokemon species seen.
        """
        binary = self._read(RamLocation.BADGES).bit_count()
        return [int(i < binary) for i in range(8)]

    @cproperty
    def event_flags(self) -> np.ndarray:
        """
        Returns the array of flag bits.
        """
        # TODO: check if we can handle this faster.
        event_bytes = np.array(self._read(slice(RamLocation.EVENT_START, RamLocation.EVENT_END, 1)), dtype=np.uint8)
        return np.unpackbits(event_bytes, bitorder="little")[EVENT_ADDRESS_ARRAY]

    @cproperty
    def event_flag_count(self) -> np.ndarray:
        """
        Returns the number of flags triggered.
        """
        return self.event_flags.sum()

    @cproperty
    def pos_x(self) -> int:
        """
        x position of the player
        """
        return self._read(RamLocation.POS_X)

    @cproperty
    def pos_y(self) -> int:
        """
        y position of the player
        """
        return self._read(RamLocation.POS_Y)

    @cproperty
    def scaled_coordinates(self) -> Tuple[float, float]:
        # 2 elements
        map_w, map_h = MapDimensions[self.map].shape

        return self.pos_x / map_h, self.pos_y / map_w


    @cproperty
    def ordered_bag_items(self) -> List[BagItem]:
        """
        Returns the bag content, ordered.
        """
        items = []
        for address in range(RamLocation.BAG_ITEMS_START, RamLocation.BAG_ITEMS_END, 2):
            item_type = BagItem(self._read(address))
            if item_type == BagItem.NO_ITEM:
                break
            items.append(item_type)
        return items + [BagItem.NO_ITEM] * (20 - len(items))

    @cproperty
    def ordered_bag_counts(self) -> List[int]:
        """
        Returns the bag content counts, ordered.
        """
        item_counts = []
        for address in range(RamLocation.BAG_ITEMS_START+1, RamLocation.BAG_ITEMS_END, 2):
            item_count = self._read(address)
            if item_count == 0:
                break
            item_counts.append(item_count)
        return item_counts + [BagItem.NO_ITEM] * (20 - len(item_counts))


    @cproperty
    def bag_items(self) -> Dict[BagItem, int]:
        """
        Returns the bag as a dict of items: counts.
        """
        bag_items = defaultdict(int) # default dict to easily add new items when buying.
        for item, count in zip(self.ordered_bag_items, self.ordered_bag_counts):
            if item == BagItem.NO_ITEM:
                break
            bag_items[item] = count

        return bag_items

    @cproperty
    def bag_count(self) -> bool:
        """
        Is the bag full M
        """
        return self._read(RamLocation.BAG_COUNT)

    @cproperty
    def player_orientation(self) -> Orientation:
        """
        Current orientation of the player sprite.
        """
        return Orientation(self._read(RamLocation.ORIENTATION))

    @cproperty
    def mart_items(self) -> List[BagItem]:
        """
        Items of the mart currently interacting with.
        """
        x, y = get_looking_at_coords(self, absolute=True, distance=2)
        mart_identifier = f"{self.map.name}@{x},{y}"
        return MartInfos.get(mart_identifier, MartInfo()).items

    @cproperty
    def can_use_cut(self) -> bool:
        """
        We can use cut only if we beat misty and have HM cut in our bag
        """
        return self.badges[Badges.CASCADE] == 1 and BagItem.HM_CUT in self.bag_items

    @cproperty
    def can_use_surf(self) -> bool:
        """
        We can use surf only if we beat Koga and have HM surf in our bag
        """
        return self.badges[Badges.SOUL] == 1 and BagItem.HM_SURF in self.bag_items

    @cproperty
    def can_use_silph_scope(self) -> bool:
        """
        We can use the silph scope only if we can use cut and have the scope in our bag
        """
        return self.can_use_cut and BagItem.SILPH_SCOPE in self.bag_items

    @cproperty
    def can_use_flute(self) -> bool:
        """
        We can use the flute only if we can use cut and have the flute in our bag
        """
        return self.can_use_cut and BagItem.POKE_FLUTE in self.bag_items

    @cproperty
    def field_moves(self) -> list:
        """
        List of available field moves
        """
        return [
            int(field_move) for field_move in [self.can_use_cut, self.can_use_flute,
                                               self.can_use_silph_scope, self.can_use_surf]
        ]

    @cproperty
    def hms(self) -> list:
        """
        Returns the vector of collected HMs.
        We do not need Flash and Strength (if we patch the game).
        # TODO. Fly does not appear to be usable for now as well. Maybe we can improve that
        """
        return [int(item in self.bag_items) for item in [
            BagItem.HM_CUT, BagItem.HM_SURF # BagItem.HM_FLY, BagItem.HM_FLASH, BagItem.HM_STRENGTH
        ]]

    @cproperty
    def safari_steps(self) -> int:
        """
        The number of steps taken in safari.
        """
        return self._read(RamLocation.SAFARI_STEPS)

    @cproperty
    def bottom_left_screen_tiles(self) -> np.ndarray:
        """
        :return: values of the bottom left tiles.
        """
        screen_tiles = self._ram_helper._get_screen_background_tilemap()
        return screen_tiles[1:1 + screen_tiles.shape[0]:2, ::2] - 256

    @cproperty
    def bottom_right_screen_tiles(self) -> np.ndarray:
        """
        TODO: UNUSED
        :return: values of the bottom right tiles.
        """
        screen_tiles = self._ram_helper._get_screen_background_tilemap()
        return screen_tiles[1:1 + screen_tiles.shape[0]:2, 1::2] - 256

    @cproperty
    def tileset_id(self) -> Union[TileSet, int]:
        tileset = self._read(RamLocation.TILESET)
        try:
            return TileSet(self._read(RamLocation.TILESET))
        except:
            return tileset

    @cproperty
    def hdst_map(self) -> int:
        """
        The state of the hdst map.
        """
        return self._read(RamLocation.HDST_MAP)

    @cproperty
    def num_warps(self) -> int:
        """
        The number of warps
        """
        return self._read(RamLocation.NUM_WARPS)

    @cproperty
    def standing_on_warp(self) -> bool:
        """
        :return: True if on warp (a hole is a warp).
        """
        return self._read(RamLocation.STANDING_ON_WARP_HOLE) == 1

    @cproperty
    def is_warping(self) -> bool:
        """
        :return: True if warping.
        """
        if (
            (
                (self._read(RamLocation.WD736) & 2**2) > 0
                and
                self.hdst_map in (255, self.map)
            )
            or
            self.standing_on_warp
        ):
            return True

        x, y = self.pos_x, self.pos_y
        for i in range(self.num_warps):
            warp_addr = RamLocation.WARP_ENTRIES + i * DataStructDimension.WARP
            if self._read(warp_addr) == y and self._read(warp_addr + 1) == x:
                return self.hdst_map in (255, self.map)

    @cproperty
    def visited_tiles(self) -> np.ndarray:
        """
        A visualisation of visited tiles around the player.
        """
        # TODO: clarify whats going on here.

        visited_tiles_on_current_map = np.zeros((9, 10), dtype=np.uint8)
        map_w, map_h = MapDimensions[self.map].shape
        pos_x, pos_y = self.pos_x, self.pos_y

        cur_top_left_x = pos_x - 4
        cur_top_left_y = pos_y - 4
        cur_bottom_right_x = pos_x + 6
        cur_bottom_right_y = pos_y + 5
        top_left_x = max(0, cur_top_left_x)
        top_left_y = max(0, cur_top_left_y)
        bottom_right_x = min(map_w, cur_bottom_right_x)
        bottom_right_y = min(map_h, cur_bottom_right_y)

        adjust_x = 0
        adjust_y = 0
        if cur_top_left_x < 0:
            adjust_x = -cur_top_left_x
        if cur_top_left_y < 0:
            adjust_y = -cur_top_left_y

        visited = self._additional_memory.visited_tiles.get(self)[top_left_y: bottom_right_y, top_left_x:bottom_right_x]
        cur_uint8_flag_count = round(255 * self.event_flag_count / len(ProgressionFlag))
        d = cur_uint8_flag_count - visited
        observed = (255 - d) * np.int32(visited > 0)

        try:
            visited_tiles_on_current_map[
            adjust_y: adjust_y + bottom_right_y - top_left_y, adjust_x: adjust_x + bottom_right_x - top_left_x
            ] = observed
        except:
            print(MapDimensions[self.map], self.pos_x, self.pos_y)

        return visited_tiles_on_current_map

    @cproperty
    def sprite_map(self) -> np.ndarray:
        """
        A minimap displaying the sprites.
        """
        # TODO: do not know the max value for s.tiles[0].tile_identifier.
        #   The game appears to have 72 sprites...

        sprite_map = np.zeros((9, 10), dtype=np.uint16)
        sprites = self._ram_helper._sprites_on_screen()
        for idx, s in enumerate(sprites):
            if (idx + 1) % 4 != 0:
                continue
            sprite_map[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4

        if self.map == Map.VERMILION_GYM and self.event_flags[EventFlag.SECOND_LOCK_OPENED] == 0:
            trashcans_coords = [
                (1, 7), (1, 9), (1, 11),
                (3, 7), (3, 9), (3, 11),
                (5, 7), (5, 9), (5, 11),
                (7, 7), (7, 9), (7, 11),
                (9, 7), (9, 9), (9, 11),
            ]
            can = self._read(RamLocation.SECOND_LOCK_TRASH_CAN) \
                if (self.event_flags[EventFlag.FIRST_LOCK_OPENED] == 1) \
                else self._read(RamLocation.FIRST_LOCK_TRASH_CAN)
            assign_new_sprite_in_sprite_minimap(sprite_map, 384, *trashcans_coords[can])

        elif self.map in switch_coords:
            for coords in switch_coords[self.map]:
                assign_new_sprite_in_sprite_minimap(sprite_map, 383, *coords)

        return sprite_map

    @cproperty
    def warp_map(self) -> np.ndarray:
        """
        A map of visible warps.F
        # Warps have their own id, are embeddings really necessary ?
        """
        warp_map = np.zeros((9, 10), dtype=np.uint16)
        warps = MapWarps[self.map]
        if len(warps) == 0:
            return warp_map

        map_w, map_h = MapDimensions[self.map].shape
        pos_x, pos_y = self.pos_x, self.pos_y

        top_left_x = max(0, pos_x - 4)
        top_left_y = max(0, pos_y - 4)
        bottom_right_x = min(map_w, pos_x + 5)
        bottom_right_y = min(map_h, pos_y + 4)

        if self.map in [Map.ROCKET_HIDEOUT_ELEVATOR, Map.SILPH_CO_ELEVATOR]:
            warps = []
            for i in range(self.num_warps):
                warp_addr = RamLocation.WARP_ENTRIES + i * DataStructDimension.WARP
                warp_y = self._read(warp_addr + 0)
                warp_x = self._read(warp_addr + 1)
                warp_id = self._read(warp_addr + 2)
                warp_map_destination = Map(self._read(warp_addr + 3))

                warps.append(MapWarp(x=warp_x, y=warp_y, id=warp_id, destination=warp_map_destination))

                # TODO: This seem to be used in rewards.
                #if warp_map_id in [199, 200, 201, 202] and warp_map_id not in self._additional_memory.hideout_elevator_maps:
                #    self.hideout_elevator_maps.append(warp_map_id)

        for warp in warps:
            if top_left_x <= warp.x <= bottom_right_x and top_left_y <= warp.y <= bottom_right_y:
                if warp.destination == Map.UNKNOWN:
                    last_map_id = self._read(RamLocation.LAST_MAP)
                    destination = Map(last_map_id)
                else:
                    destination = warp.destination


                # TODO: take care of unknown warps.
                encoded_warp = NamedWarpIds[f"{destination.name}@{warp.id - 1}"]
                warp_map[warp.y - top_left_y, warp.x - top_left_x] = encoded_warp + 1

        return warp_map

    @cproperty
    def feature_maps(self) -> np.ndarray:
        """
        A set of map features compiled in one image.
        1. walkable tiles
        2. trees
        3. overworld ledges
        4. water
        5. previously visited tiles
        """
        maps = np.zeros((7, 9, 10), dtype=np.uint8)
        # TODO:
        #    clean up this function, we do not understand whats going on very well.
        #   - note: they do not update the count when player is warping ?
        # TODO: how do they process this into the model after ??

        bottom_left_screen_tiles = self.bottom_left_screen_tiles
        # walkable
        maps[0] = self._ram_helper._get_screen_walkable_matrix() * 255

        # Water
        if self.tileset_id == TileSet.VERMILION_PORT:
            maps[5] = (bottom_left_screen_tiles == 20).astype(np.uint8) * 255
        elif self.tileset_id in [TileSet.OVERWORLD, TileSet.FOREST, TileSet.TILESET_5, TileSet.GYM, TileSet.TILESET_13,
                                 TileSet.TILESET_17, TileSet.TILESET_22, TileSet.TILESET_23]:
            maps[5] = np.isin(bottom_left_screen_tiles, [0x14, 0x32, 0x48]).astype(np.uint8) * 255

        if self.tileset_id == TileSet.OVERWORLD:  # is overworld
            # tree
            maps[1] = (bottom_left_screen_tiles == 61).astype(np.uint8) * 255
            # ledge down
            maps[2] = np.isin(bottom_left_screen_tiles, [54, 55]).astype(np.uint8) * 255
            # ledge left
            maps[3] = (bottom_left_screen_tiles == 39).astype(np.uint8) * 255
            # ledge right
            maps[4] = np.isin(bottom_left_screen_tiles, [13, 29]).astype(np.uint8) * 255
        elif self.tileset_id == TileSet.GYM:
            # tree
            maps[1] = (bottom_left_screen_tiles == 80).astype(np.uint8) * 255 # 0x50

        maps[6] = self.visited_tiles

        return maps
    
    @cproperty
    def box_pokemon_stats(self) -> List[PokemonStats]:
        """
        List of stats of the box pokemons
        """
        box_pokemon_stats = []
        for i in range(self.box_pokemon_count):
            offset = RamLocation.BOX_POKEMON_START + i * DataStructDimension.BOX_POKEMON_STATS
            species = self._read(RamLocation.BOX_POKEMON_SPECIES_START + i)
            level = self._read(offset + 3)
            exp = self._read_triple(offset + 14)
            hp_ev = self._read_double(offset + 17)
            atk_ev = self._read_double(offset + 19)
            def_ev = self._read_double(offset + 21)
            spd_ev = self._read_double(offset + 23)
            spc_ev = self._read_double(offset + 25)
            atk_def_iv = self._read(offset + 27)
            spd_spc_iv = self._read(offset + 28)

            atk_iv = atk_def_iv >> 4
            def_iv = atk_def_iv & 0xF
            spd_iv = spd_spc_iv >> 4
            spc_iv = spd_spc_iv & 0xF

            hp_iv = 0
            hp_iv += 8 if atk_iv % 2 == 1 else 0
            hp_iv += 4 if def_iv % 2 == 1 else 0
            hp_iv += 2 if spd_iv % 2 == 1 else 0
            hp_iv += 1 if spc_iv % 2 == 1 else 0

            pokemon_stats = PokemonStats(
                pokemon=Pokemon(species),
                level=level,
                exp=exp,
                evs=PokemonBaseStats(
                    hp=hp_ev, attack=atk_ev, defense=def_ev, speed=spd_ev, special=spc_ev
                ),
                ivs=PokemonBaseStats(
                    hp=hp_iv, attack=atk_iv, defense=def_iv, speed=spd_iv, special=spc_iv

                )
            )
            box_pokemon_stats.append(pokemon_stats)
            
        return box_pokemon_stats

    @cproperty
    def focused_pokemon(self) -> int:
        return self._read(RamLocation.WHICH_POKEMON)

    @cproperty
    def pokemon_move_learning_powers(self) -> Tuple[List[float], float]:
        """
        Returns the move powers of the pokemon currently learning a move
        """
        ptypes = [self._read(RamLocation.PARTY_0_TYPE0 + (self.focused_pokemon * DataStructDimension.POKEMON_STATS) + i)
                  for i in range(2)]
        move_powers = [
            MovesInfo[Move(self._read(RamLocation.WHICH_POKEMON_LEARNED_MOVES + i))].actual_power(ptypes)
            for i in range(4)]
        new_move_power = MovesInfo[Move(self._read(RamLocation.MOVE_TO_LEARN))].actual_power(ptypes)
        return move_powers, new_move_power

    @cproperty
    def box_pokemon_stat_sums(self) -> List[int]:
        """
        List of stat sums of the box pokemons
        """
        return [box_pokemon.scale(level=stats_level_scale).sum() for box_pokemon in self.box_pokemon_stats]

    @cproperty
    def best_box_pokemon_index(self) -> int:
        """
        Index of the pokemon with the highest stat sum in the box
        """
        print(self.box_pokemon_stat_sums, self.party_pokemon_stat_sums)
        return 0 if self.box_pokemon_count == 0 else int(np.argmax(self.box_pokemon_stat_sums))
        
    @cproperty
    def party_pokemon_stat_sums(self) -> List[int]:
        """
        List of stat sums of the party
        """
        party_stat_sums = []
        for i in range(self.party_count):
            level = self._read(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 33)
            hp = self._read_double(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 34)
            attack = self._read_double(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 36)
            defense = self._read_double(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 38)
            speed = self._read_double(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 40)
            special = self._read_double(RamLocation.PARTY_START + i * DataStructDimension.POKEMON_STATS + 42)
            party_stat_sums.append(
                PokemonBaseStats(hp=hp, attack=attack, defense=defense, speed=speed, special=special).rescale_as(
                    original_level=level, target_level=stats_level_scale
                ).sum()
            )
        return party_stat_sums

    @cproperty
    def worst_party_pokemon_index(self) -> int:
        """
        Index of the pokemon with the lowest stat sum in the party
        """
        return int(np.argmin(self.party_pokemon_stat_sums))

    @cproperty
    def has_better_pokemon_in_box(self) -> bool:
        """
        We use the additional memory so that we do not compute every stats every step.
        """
        return self._additional_memory.good_pokemon_in_box.better_pokemon_in_box

    @cproperty
    def visited_pokemon_centers(self) -> List[int]:
        """
        Pokemon centers that have been visited
        """
        return self._additional_memory.pokecenter_checkpoints.registered_checkpoints

    @cproperty
    def last_visited_maps(self) -> List[Map]:
        """
        Latest maps that have been visited
        """
        return self._additional_memory.map_history.map_history

    @cproperty
    def last_triggered_flags(self) -> List[ProgressionFlag]:
        """
        The latest triggered flag events.
        """
        return self._additional_memory.flag_history.flag_history

    @cproperty
    def last_triggered_flags_age(self) -> List[int]:
        """
        The latest triggered flag events.
        """
        return [self.step - step for step in self._additional_memory.flag_history.stepstamps]


def pokemon_status(status_byte):
    """
    Converts a byte into status bits. Last bit unused.
    """
    return [int(status_byte & 2**i > 0) for i in range(7)]

def fix_pokemon_types(types: np.ndarray) -> np.ndarray:
    """
    TODO: check if this is fine ?
    """
    types[types>=9] -= 11
    return types


switch_coords = {
    Map.POKEMON_MANSION_1F: [(2, 5)],
    Map.POKEMON_MANSION_2F: [(2, 11)],
    Map.POKEMON_MANSION_3F: [(10, 5)],
    Map.POKEMON_MANSION_B1F: [(20, 3), (18, 25)],

}

pokecenter_pc_location = [(13, 4)]
pc_locations = {
    Map.INDIGO_PLATEAU_LOBBY: [(15, 8)],
    Map.CELADON_MANSION_2F: [(0, 6)],
    Map.SILPH_CO_11F: [(10, 13)],
    Map.CINNABAR_LAB_FOSSIL_ROOM: [(0, 5), (2, 5)],
}

def assign_new_sprite_in_sprite_minimap(
        minimap,
        sprite_id,
        x, y
):
    top_left_x = x - 4
    top_left_y = y - 4
    if x >= top_left_x and x < top_left_x + 10 and y >= top_left_y and y < top_left_y + 9:
        minimap[y - top_left_y, x - top_left_x] = sprite_id

def get_looking_at_coords(gamestate: GameState, absolute=False, distance=1) -> Tuple[int, int]:

    if absolute:
        x = gamestate.pos_x
        y = gamestate.pos_y
    else:
        # relative position w.r.t. screen center
        x = 4
        y = 4
    if gamestate.player_orientation == Orientation.LEFT:
        x -= distance
    elif gamestate.player_orientation == Orientation.RIGHT:
        x += distance
    elif gamestate.player_orientation == Orientation.UP:
        y -= distance
    elif gamestate.player_orientation == Orientation.DOWN:
        y += distance
    return x, y