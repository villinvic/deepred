from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
from PIL import Image
from pyboy import PyBoy

from deepred.polaris_env.enums import StartMenuItem, Map, RamLocation, Pokemon, BagItem, EventFlag, TileSet, \
    DataStructDimension
from deepred.polaris_env.utils import cproperty

B = 256
B2 = B**2

EVENT_ADDRESS_ARRAY = np.array([
    e.value for e in EventFlag
])

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
    def is_in_battle(self) -> bool:
        """
        :return: True if in battle
        """
        # Battle type is 0 when not in battle.
        return self.battle_type != 0

    def is_dialog_frame(self) -> bool:
        """
        :return: True if we are currently at a dialog frame. Utility for skipping dialogs.
        """
        return (
                self.start_menu_item == StartMenuItem.UNSELECTED
                and not self.is_in_battle
                and self.open_text_box
                and not self.textbox_id in {6, 11, 12, 13, 14, 20, 28}
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
            addr: Union[RamLocation, slice]
    ) -> Union[int, List]:
        """
        Reads the game's RAM at the given address as a double.
        :param addr: address (one index, or slice) where we want to read the RAM.
        :return: The memory value at the given addr as a double.
        """
        if isinstance(addr, RamLocation):
            return 256 * self._read(addr) + self._read(addr + 1)

        s1 = slice(addr.start, addr.stop, 2)
        s2 = slice(addr.start+1, addr.stop+1, 2)

        return B * self._read(s1) + self._read(s2)

    def _read_triple(
            self,
            addr: Union[RamLocation, slice]
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
            addr: Union[RamLocation, slice]
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
    def is_playing_party_animations(self) -> int:
        """
        When opening the party screen, animations for pokemons play, this reads the index of the sprite.
        :return: Returns the index of the sprite shown, 0 if not played
        """
        return self._read(RamLocation.PARTY_MENU_ANIMATION_INDEX)

    @cproperty
    def battle_type(self) -> int:
        """
        :return: True if in battle
        """
        # Battle type is 0 when not in battle.
        return self._read(RamLocation.BATTLE_TYPE)

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
    def party_pokemons(self) -> List[Pokemon]:
        """
        The current party pokemon ids.
        """
        return [Pokemon(ID) for ID in self._read(slice(RamLocation.PARTY_0_ID, RamLocation.PARTY_5_ID+1, 1))]

    @cproperty
    def party_count(self) -> int:
        """
        :return: The number of pokemons in the current party
        """
        s = 0
        for pokemon in self.party_pokemons:
            if pokemon not in (Pokemon.NONE, Pokemon.M_GLITCH):
                s += 1
        return s

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
    def start_menu_item(self) -> StartMenuItem:
        """
        :return: Start item currently selected, returns unselected if the menu is not open.
        """
        # correct_cursor_cords = self.item_cursor_x, self.item_cursor_y == (2, 11)
        # dialog_state = self._read(RamLocation.DIALOG_STATE)
        # start_menu_open = (self.map == Map.REDS_HOUSE_SECOND_FLOOR and dialog_state == 8) or dialog_state ==  113
        # TODO: needs another check for which menu we are in exactly, cursor does not reset after closing the menu
        #if self.open_text_box and correct_cursor_cords and start_menu_open:
        if self.open_start_menu:
            return StartMenuItem(self._read(RamLocation.MENU_ITEM_ID))

        return StartMenuItem.UNSELECTED

    @cproperty
    def party_hp(self) -> List[float]:
        """
        The current party pokemon health fractions (0 is fainted, 1 is full hp).
        # TODO: handle pokemon indices between game and battles (they use different rams)
        # https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Battle_2
        """
        return [ self._read_double(curr_hp_addr) / (self._read_double(max_hp_addr) + 1e-8)
        # Addresses are not adjacent.
        for curr_hp_addr, max_hp_addr in [
                     (RamLocation.PARTY_0_HP, RamLocation.PARTY_0_MAXHP),
                     (RamLocation.PARTY_1_HP, RamLocation.PARTY_1_MAXHP),
                     (RamLocation.PARTY_2_HP, RamLocation.PARTY_2_MAXHP),
                     (RamLocation.PARTY_3_HP, RamLocation.PARTY_3_MAXHP),
                     (RamLocation.PARTY_4_HP, RamLocation.PARTY_4_MAXHP),
                     (RamLocation.PARTY_5_HP, RamLocation.PARTY_5_MAXHP),
                 ]
        ]

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
        # TODO: determine what is the fastest here.
        event_bytes = np.array(self._read(slice(RamLocation.EVENT_START, RamLocation.EVENT_END, 1)), dtype=np.uint8)
        return np.unpackbits(event_bytes, bitorder="little")[EVENT_ADDRESS_ARRAY]

    @cproperty
    def pos_x(self) -> int:
        return self._read(RamLocation.POS_X)

    @cproperty
    def pos_y(self) -> int:
        return self._read(RamLocation.POS_Y)

    @cproperty
    def bag_items(self) -> Dict[BagItem, int]:
        # item_types = self._read(slice(RamLocation.BAG_ITEMS_START, RamLocation.BAG_ITEMS_END, 2))
        # item_counts = self._read(slice(RamLocation.BAG_ITEMS_START+1, RamLocation.BAG_ITEMS_END, 2))
        bag_items = {}
        for address in range(RamLocation.BAG_ITEMS_START, RamLocation.BAG_ITEMS_END, 2):
            item_type = self._read(address)
            item_count = self._read(address + 1)

            if item_type == 255:
                break

            bag_items[BagItem(item_type)] = item_count

        return bag_items

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
    def minimap_sprite(self) -> np.ndarray:
        """
        A minimap displaying the sprites.
        """
        minimap_sprite = np.zeros((9, 10), dtype=np.int16)
        sprites = self._ram_helper._sprites_on_screen()
        for idx, s in enumerate(sprites):
            if (idx + 1) % 4 != 0:
                continue
            minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4

        if self.map == Map.VERMILION_GYM and self.event_flags[EventFlag.EVENT_2ND_LOCK_OPENED] == 0:
            trashcans_coords = [
                (1, 7), (1, 9), (1, 11),
                (3, 7), (3, 9), (3, 11),
                (5, 7), (5, 9), (5, 11),
                (7, 7), (7, 9), (7, 11),
                (9, 7), (9, 9), (9, 11),
            ]
            can = self._read(RamLocation.SECOND_LOCK_TRASH_CAN) \
                if (self.event_flags[EventFlag.EVENT_1ST_LOCK_OPENED] == 1) \
                else self._read(RamLocation.FIRST_LOCK_TRASH_CAN)
            assign_new_sprite_in_sprite_minimap(minimap_sprite, 384, *trashcans_coords[can])

        elif self.map in switch_coords:
            for coords in switch_coords[self.map]:
                assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, *coords)

    @cproperty
    def feature_maps(self) -> np.ndarray:
        """
        A set of map features compiled in one image.
        """
        # TODO: seen map: dict of maps
        explored_tiles = defaultdict(
            lambda : defaultdict(int)
        )

        # TODO: do we post process by normalising ?
        explored_tiles[self.map][(self.pos_x, self.pos_y)] = self.frame

        ledges_dict = {
            'down': [54, 55],
            'left': 39,
            'right': [13, 29]
        }
        minimap = np.zeros((6, 9, 10), dtype=np.float32)
        bottom_left_screen_tiles = self.bottom_left_screen_tiles
        # walkable
        minimap[0] = self._ram_helper._get_screen_walkable_matrix()

        if self.tileset_id == TileSet.VERMILION_PORT:
            minimap[5] = (bottom_left_screen_tiles == 20).astype(np.float32)
        elif self.tileset_id in [TileSet.OVERWORLD, TileSet.FOREST, TileSet.TILESET_5, TileSet.GYM, TileSet.TILESET_13,
                                 TileSet.TILESET_17, TileSet.TILESET_22, TileSet.TILESET_23]:
            # What is this ? water ?
            minimap[5] = np.isin(bottom_left_screen_tiles, [0x14, 0x32, 0x48]).astype(np.float32)

        if self.tileset_id == TileSet.OVERWORLD:  # is overworld
            # tree
            minimap[1] = (bottom_left_screen_tiles == 61).astype(np.float32)
            # ledge down
            minimap[2] = np.isin(bottom_left_screen_tiles, ledges_dict['down']).astype(np.float32)
            # ledge left
            minimap[3] = (bottom_left_screen_tiles == ledges_dict['left']).astype(np.float32)
            # ledge right
            minimap[4] = np.isin(bottom_left_screen_tiles, ledges_dict['right']).astype(np.float32)
        elif self.tileset_id == TileSet.GYM:
            # tree
            minimap[1] = (bottom_left_screen_tiles == 80).astype(np.float32)  # 0x50

        # get seen_map obs
        seen_map_obs = self.get_all_seen_map_obs()  # (8, 9, 10)

        minimap = np.concatenate([minimap, seen_map_obs], axis=0)  # (14, 9, 10)
        self._minimap_obs = minimap
        return self._minimap_obs


switch_coords = {
    Map.POKEMON_MANSION_1F: [(2, 5)],
    Map.POKEMON_MANSION_2F: [(2, 11)],
    Map.POKEMON_MANSION_3F: [(10, 5)],
    Map.POKEMON_MANSION_B1F: [(20, 3), (18, 25)],

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