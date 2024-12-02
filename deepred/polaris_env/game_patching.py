import inspect
import sys
from typing import List, Tuple, Type

import numpy as np
from tensorflow.python.types.core import Callable

from deepred.polaris_env.enums import RamLocation, DataStructDimension, Map, BagItem, EventFlag
from deepred.polaris_env.gamestate import GameState


# TODO:
#   -game_path
#   -items:
#       -buy
#       -replace in bag


def to_bcd(amount):
    """
    :param amount: amount to convert into bcd
    :return: the bcd amount.
    """
    return ((amount // 10) << 4) + (amount % 10)

def set_bit(
        byte: int,
        bit: int
):
    """
    Sets the bit of the byte to 1.
    :return: the new byte.
    """
    return byte | (1<<bit)

def set_player_money(
        ram,
        gamestate: GameState,
        amount: int,
):
    """
    Sets the player money to the specified amount
    """
    amount = np.clip(amount, 0, 999999)
    ram[RamLocation.MONEY3] = to_bcd(amount) // 10000
    ram[RamLocation.MONEY2] = to_bcd((amount % 10000) // 100)
    ram[RamLocation.MONEY1] = to_bcd(amount % 100)


def out_of_cash_safari_patch(
        ram,
        gamestate: GameState
):
    """
    Fixes the minimum cash to 500 if going into safari with small amount of money (<501).
    """
    if gamestate.map == Map.SAFARI_ZONE_GATE:
        if gamestate.player_money < 501:
            set_player_money(
                ram,
                gamestate,
                500
            )

def infinite_time_safari_patch(
        ram,
        gamestate: GameState
):
    """
    Sets the safari timer to infinite if we have:
        - Surf and Strength
        - Surf and Gold teeth
    """
    if (
        not Map.SAFARI_ZONE_EAST <=gamestate.map<= Map.SAFARI_ZONE_NORTH_REST_HOUSE
        or
        (
            BagItem.HM_SURF in gamestate.bag_items
            and
            (
                BagItem.GOLD_TEETH in gamestate.bag_items
                or
                BagItem.HM_STRENGTH in gamestate.bag_items
            )
        )
        or
        gamestate.safari_steps > 0

    ):
        return

    ram[RamLocation.SAFARI_STEPS] = 1

def instantaneous_text_patch(
        ram,
        gamestate: GameState,
):
    """
    Sets the text diplay to instantaneous.
    """
    if gamestate.instant_text:
        return
    ram[RamLocation.INSTANT_TEXT] = set_bit(ram[RamLocation.INSTANT_TEXT], 6)


def nerf_spinners_path(
        ram,
        gamestate: GameState,
):
    """
    Changes the spinner behaviors, to make the player move only one tile back.
    """

    if gamestate.map in [Map.VIRIDIAN_GYM, Map.ROCKET_HIDEOUT_B1F, Map.ROCKET_HIDEOUT_B2F,
                         Map.ROCKET_HIDEOUT_B3F, Map.ROCKET_HIDEOUT_B4F]:
        ram[RamLocation.SIMULATED_JOYPAD_INDEX] = 0

def victory_road_patch(
        ram,
        gamestate: GameState
):
    """
    Clears the boulder puzzle.
    """
    if gamestate.map not in [Map.VICTORY_ROAD_1F, Map.VICTORY_ROAD_2F, Map.VICTORY_ROAD_3F, Map.ROUTE_23]:
        return

    for address, bit in [
        [0xD7EE, 0], # EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH1
        [0xD7EE, 7],
        [0xD813, 0], # EVENT_VICTORY_ROAD_3_BOULDER_ON_SWITCH1
        [0xD813, 6],
        [0xD869, 7], # EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH
    ]:
        ram[address] = set_bit(ram[address], bit)


def elevator_patch(
        ram,
        gamestate: GameState
):
    """
    Automates elevators
    """
    if (
        gamestate.map != Map.ROCKET_HIDEOUT_ELEVATOR
        and
        BagItem.LIFT_KEY in gamestate.bag_items
    ):
        warp_map, warp_id = (Map.ROCKET_HIDEOUT_B1F, 4) if BagItem.SILPH_SCOPE in gamestate.bag_items \
            else (Map.ROCKET_HIDEOUT_B4F, 2)

    elif gamestate.map == Map.SILPH_CO_ELEVATOR:
        # no key_card, lift exit warp = 5f
        # have key_card, lift exit warp = 3f
        # have key_card, cleared masterball event, = 1f?
        if BagItem.CARD_KEY not in gamestate.bag_items:
            warp_map, warp_id = (Map.SILPH_CO_5F, 2)
        elif gamestate.event_flags[EventFlag.EVENT_GOT_MASTER_BALL]:
            warp_map, warp_id = (Map.SILPH_CO_1F, 3)
        else:
            warp_map, warp_id = (Map.SILPH_CO_3F, 2)
    else:
        return

    # Update warps to automate the elevators.
    ram[RamLocation.WARP_ENTRIES + 2] = warp_id
    ram[RamLocation.WARP_ENTRIES + 3] = warp_map
    ram[RamLocation.WARP_ENTRIES + 2 + DataStructDimension.WARP] = warp_id
    ram[RamLocation.WARP_ENTRIES + 3 + DataStructDimension.WARP] = warp_map


patches = {
    name[:-6]: obj for name, obj in globals().items()
                   if callable(obj) and name.endswith("_patch")
}
"""
Patches:
- allow safari whenever out of cash
- infinite safari time
- instantaneous text
- nerf spinners
- victory road puzzle clear
- automated elevators
"""


class GamePatching:
    def __init__(
            self,
            enabled_patches: Tuple[str] = tuple(patches.keys()),
    ):
        self.patches = []
        for patch in enabled_patches:
            if patch not in patches:
                print(f"Unknown patch {patch}.")
                continue
            self.patches.append(patches[patch])

    def patch(
            self,
            gamestate: GameState
    ):
        """
        Should be run with each environment step.
        Runs a list of functions over the gamestate to:
            - to alter the game's ram.
            - allow actions that are normally done by humans but hard for agents.
        :param gamestate: The gamestate we use to update the ram (accessed through )
        """
        for patch in self.patches:
            patch(
                gamestate._ram,
                gamestate
            )


class AgentHelper:

    """

    - Auto switch
    - Auto learn better moves
    - routines for HMs, flute
    - manage pokemons in party, box
    - manage bag items
    - sell/buy items
    - wait whenever an action can be taken (I think the modded rom takes care of this already).
    """

    def roll_party(
            self,
            ram,
            gamestate: GameState
    ):
        """
        Helper action/function for the agent.
        We roll the party pokemons one slot.
        """

        # Do not roll if we do not have enough pokemons in the party, if we are in battle, or if a menu is open.
        if (
                gamestate.party_count == 1
                or
                gamestate.is_in_battle
                or
                gamestate.open_menu
        ):
            return

        stats_addresses = slice(
            RamLocation.PARTY_START,
            RamLocation.PARTY_START + DataStructDimension.POKEMON_STATS,
            1
        )

        nickname_addresses = slice(
            RamLocation.PARTY_NICKNAMES_START,
            RamLocation.PARTY_NICKNAMES_START + DataStructDimension.POKEMON_NICKNAME,
            1
        )

        # the pokemon at slot 0 will wrap around.
        slot_0 = {
            "species": ram[RamLocation.PARTY_0_ID],
            "stats": ram[stats_addresses],
            "nickname": ram[nickname_addresses]
        }

        for slot in range(gamestate.party_count - 1):
            # move each pokemon by one slot.
            # species
            ram[RamLocation.PARTY_0_ID + slot] = ram[RamLocation.PARTY_0_ID + (slot + 1)]
            ram[RamLocation.PARTY_0_ID + slot] = ram[RamLocation.PARTY_0_ID + (slot + 1)]

            next_stats_addresses = slice(
                RamLocation.PARTY_START + (slot + 1) * DataStructDimension.POKEMON_STATS,
                RamLocation.PARTY_START + (slot + 2) * DataStructDimension.POKEMON_STATS,
                1
            )
            ram[stats_addresses] = next_stats_addresses
            stats_addresses = next_stats_addresses

            next_nickname_addresses = slice(
                RamLocation.PARTY_NICKNAMES_START + (slot + 1) * DataStructDimension.POKEMON_NICKNAME,
                RamLocation.PARTY_NICKNAMES_START + (slot + 2) * DataStructDimension.POKEMON_NICKNAME,
                1
            )
            ram[nickname_addresses] = next_nickname_addresses
            nickname_addresses = next_nickname_addresses

        ram[RamLocation.PARTY_0_ID + (gamestate.party_count - 1)] = slot_0["species"]
        ram[stats_addresses] = slot_0["stats"]
        ram[nickname_addresses] = slot_0["nickname"]


    '''
def party_stat_sum():
    pass

def box_stat_sum():
    pass

def move_best_box_pokemon_to_party(
        ram,
        gamestate: GameState
):
    """
    Moves the lowest total stats pokemon in the party with the highest stats pokemon in the box.

    If the party is not full, just add the best pokemon.
    If the box is empty, do nothing.
    """
    # if no pokemon in box, do nothing
    if gamestate.box_pokemon_count == 0:
        return

    box_stat_sums = []
    best_box_pokemon = np.argmax(box_stat_sums)
    # If our party is not full, just move the best box pokemon into our party
    if gamestate.box_pokemon_count < 6:
        return move_box_pokemon_to_party(best_box_pokemon)

    party_stat_sums = []
    worst_party_pokemon = np.argmin(party_stat_sums)

    if party_stat_sums[worst_party_pokemon] > box_stat_sums[best_box_pokemon]:
        # We found a pokemon that is worse that our team in the box, just release worst pokemons.
        return release_weak_pokemons(stat_sum_threshold=party_stat_sums[worst_party_pokemon])

    # box pokemon
    box_stats_dict = {}
    index = highest_box_level_idx

    lowest_party_species = self.read_m(party_species_addr_start + lowest_party_level_idx)
    highest_box_species = self.read_m(box_species_addr_start + highest_box_level_idx)
    if lowest_party_species in ID_TO_SPECIES and highest_box_species in ID_TO_SPECIES:
        print(
            f'\nSwapping pokemon {ID_TO_SPECIES[lowest_party_species]} lv {lowest_party_level} with {ID_TO_SPECIES[highest_box_species]} lv {highest_box_level}')
    else:
        print(
            f'\nSwapping pokemon {lowest_party_species} lv {lowest_party_level} with {highest_box_species} lv {highest_box_level}')
    self.use_pc_swap_count += 1
    # calculate box pokemon stats
    box_stats_dict = self.calculate_pokemon_stats(box_stats_dict)

    # swap party pokemon with box pokemon
    # copy species
    self.pyboy.set_memory_value(party_species_addr_start + lowest_party_level_idx,
                                self.read_m(box_species_addr_start + highest_box_level_idx))

    # copy all 0 to 33 from box to party
    for i in range(33):
        self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + i,
                                    self.read_m(box_mon_addr_start + highest_box_level_idx * 33 + i))
        if i == 3:
            # copy level from box to party
            self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 33,
                                        self.read_m(box_mon_addr_start + highest_box_level_idx * 33 + 3))

    # copy the remaining stats from box to party
    # max_hp, atk, def, spd, spc
    box_stats = [box_stats_dict['max_hp'], box_stats_dict['atk'], box_stats_dict['def'], box_stats_dict['spd'],
                 box_stats_dict['spc']]
    for i in range(5):
        # these stats are splitted into 2 bytes
        # first byte is the higher byte, second byte is the lower byte
        self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 34 + i * 2, box_stats[i] >> 8)
        self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 35 + i * 2,
                                    box_stats[i] & 0xFF)

    # copy nickname
    for i in range(11):
        self.pyboy.set_memory_value(party_nicknames_addr_start + lowest_party_level_idx * 11 + i,
                                    self.read_m(box_nicknames_addr_start + highest_box_level_idx * 11 + i))

    self.delete_box_pokemon(highest_box_level_idx, num_mon_in_box)
    self._num_mon_in_box = None

    self.delete_box_pokemon_with_low_level(lowest_party_level)

def delete_box_pokemon_with_low_level(self, lowest_party_level):
    box_mon_addr_start = 0xda96
    # delete all box pokemon with level < party lowest level
    # step 1 find all box pokemon with level < party lowest level
    # step 2 delete them
    # step 3 update num_mon_in_box
    num_mon_in_box = self.num_mon_in_box
    if num_mon_in_box == 0:
        # no pokemon in box, do nothing
        return
    box_levels = [self.read_m(box_mon_addr_start + i * 33 + 3) for i in range(num_mon_in_box)]
    box_levels_to_delete = [i for i, x in enumerate(box_levels) if x <= lowest_party_level]
    # start from the last index to delete
    for i in range(len(box_levels_to_delete) - 1, -1, -1):
        self.delete_box_pokemon(box_levels_to_delete[i], num_mon_in_box)
        self._num_mon_in_box = None
        num_mon_in_box = self.num_mon_in_box

def delete_box_pokemon(self, box_mon_idx, num_mon_in_box):
    box_mon_addr_start = 0xda96
    box_species_addr_start = 0xDA81
    box_nicknames_addr_start = 0xde06
    # delete the box pokemon by shifting the rest up
    # box mon only has 33 stats
    for i in range(box_mon_idx, num_mon_in_box - 1):
        # species
        self.pyboy.set_memory_value(box_species_addr_start + i, self.read_m(box_species_addr_start + i + 1))
        # stats
        for j in range(33):
            self.pyboy.set_memory_value(box_mon_addr_start + i * 33 + j,
                                        self.read_m(box_mon_addr_start + (i + 1) * 33 + j))
        # nickname
        for j in range(11):
            self.pyboy.set_memory_value(box_nicknames_addr_start + i * 11 + j,
                                        self.read_m(box_nicknames_addr_start + (i + 1) * 11 + j))

    # reduce num_mon_in_box by 1
    self.pyboy.set_memory_value(0xda80, num_mon_in_box - 1)
    # set the last box pokemon species to ff as it is empty
    self.pyboy.set_memory_value(box_species_addr_start + (num_mon_in_box - 1), 0xff)
    # set the last box pokemon stats to 0
    for i in range(33):
        self.pyboy.set_memory_value(box_mon_addr_start + (num_mon_in_box - 1) * 33 + i, 0)
    # set the last box pokemon nickname to 0
    for i in range(11):
        self.pyboy.set_memory_value(box_nicknames_addr_start + (num_mon_in_box - 1) * 11 + i, 0)

    '''