from pathlib import Path
from typing import Union, List, Dict

import numpy as np
from PIL import Image

from deepred.polaris_env.enums import StartMenuItem, Map, RamLocation, Pokemon, BagItem, EventFlag
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

        self._frame = console._frame
        self.screen = np.uint8(console.screen.ndarray[:, :, 0])


        # we need to fetch those states no matter what. We need them for skipping frames
        self.start_menu_item
        self.is_in_battle()

    def is_skippable_frame(self) -> bool:
        """
        :return: Whether this frame can be skipped.
        """
        return self.is_skippable_textbox() or self.is_transition_screen()

    def is_skippable_textbox(self) -> bool:
        """
        :return: True if we can skip the textbox by ticking
        """
        # TODO: find reliable way to skip dialogs...
        return self.is_skippable_battle_frame() #or self.is_dialog_frame()


    def is_skippable_battle_frame(self):
        # We skip the frame if we are in battle, have a certain box id open or if we are in the party menu.
        return (
                self.is_in_battle()
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
                and not self.is_in_battle()
                and self.open_text_box
                and not self.textbox_id in {6, 11, 12, 13, 14, 20}
                and not self.is_playing_party_animations == 64
                and not self.open_town_map
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
            Map.POKEMON_CENTER_ROUTE_4, Map.POKEMON_CENTER_PEWTER_CITY, Map.DAYCARE_CENTER_ROUTE_5,
            Map.POKEMON_CENTER_CERULEAN_CITY, Map.POKEMON_CENTER_VIRIDIAN_CITY, Map.POKEMON_CENTER_CELADON_CITY,
            Map.POKEMON_CENTER_ROCK_TUNNEL, Map.POKEMON_CENTER_LAVENDER_TOWN, Map.POKEMON_CENTER_FUCHSIA_CITY,
            Map.POKEMON_CENTER_CINNABAR_ISLAND, Map.POKEMON_CENTER_SAFFRON_CITY, Map.POKEMON_CENTER_VERMILION_CITY,
            Map.POKEMON_CENTER_INDIGO_PLATEAU
        ]

    def is_at_pokemart(self) -> bool:
        """
        :return: True if we are at a pokemart.
        """
        return self.map in [
            Map.POKE_MART_PEWTER_CITY, Map.POKE_MART_FUCHSIA_CITY, Map.POKE_MART_SAFFRON_CITY,
            Map.POKE_MART_CERULEAN_CITY, Map.POKE_MART_VIRIDIAN_CITY, Map.POKE_MART_VERMILION_CITY,
            Map.POKE_MART_CINNABAR_ISLAND, Map.POKE_MART_LAVENDER_TOWN, Map.POKE_MART__ALTERNATIVE_MUSIC_CINNABAR_ISLAND
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
    def textbox_id(self) -> int:
        return self._read(RamLocation.TEXTBOX_ID)

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
        return self._read(RamLocation.OPEN_TOWN_MAP) == 255

    @cproperty
    def party_pokemons(self) -> List[Pokemon]:
        """
        The current party pokemon ids.
        """
        return [Pokemon(ID) for ID in self._read(slice(RamLocation.PARTY_0_ID, RamLocation.PARTY_5_ID+1, 1))]

    @cproperty
    def open_text_box(self) -> bool:
        """
        :return: True if a textbox is currently open
        """
        return self._read(RamLocation.OPEN_TEXT_BOX) == 1

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
    def dialog_state(self) -> int:
        """
        :return: Returns the y coordinates (on the screen) of the topmost item cursor
        """
        return self._read(RamLocation.DIALOG_STATE)

    @cproperty
    def start_menu_item(self) -> StartMenuItem:
        """
        :return: Start item currently selected, returns unselected if the menu is not open.
        """
        correct_cursor_cords = self.item_cursor_x, self.item_cursor_y == (2, 11)
        dialog_state = self._read(RamLocation.DIALOG_STATE)
        start_menu_open = (self.map == Map.REDS_HOUSE_SECOND_FLOOR and dialog_state == 8) or dialog_state ==  113
        # TODO: needs another check for which menu we are in exactly, cursor does not reset after closing the menu
        if self.open_text_box and correct_cursor_cords and start_menu_open:
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
