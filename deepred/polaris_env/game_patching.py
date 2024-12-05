from typing import Tuple

import numpy as np

from deepred.polaris_env.pokemon_red.enums import RamLocation, DataStructDimension, Map, BagItem, EventFlag
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

def to_double(amount):
    """
    returns a tuple, values for both bytes
    """
    byte1 = (amount >> 8) & 0xFF
    byte2 = amount & 0xFF
    return byte2, byte1

def to_triple(amount):
    """
    returns a tuple, values for both bytes
    """
    byte1 = (amount >> 16) & 0xFF
    byte2 = (amount >> 8) & 0xFF
    byte3 = amount & 0xFF
    return byte3, byte2, byte1

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
        if BagItem.CARD_KEY not in gamestate.bag_items:
            warp_map, warp_id = (Map.SILPH_CO_5F, 2)
        elif gamestate.event_flags[EventFlag.GOT_MASTER_BALL]:
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

def freshwater_trade_patch(
        ram,
        gamestate: GameState
):
    if gamestate.step == 1:
        ram[0xD778] = set_bit(0xD778, 4)


def seafoam_island_patch(
            ram,
            gamestate: GameState
    ):
        if gamestate.map not in (Map.SEAFOAM_ISLANDS_1F, Map.SEAFOAM_ISLANDS_B1F,
                             Map.SEAFOAM_ISLANDS_B2F, Map.SEAFOAM_ISLANDS_B3F, Map.SEAFOAM_ISLANDS_B4F):
            return

        for address, bit in [
            [0xD7E8, 6],
            [0xD7E8, 7],
            [0xD87F, 0],
            [0xD87F, 1],
            [0xD880, 0],
            [0xD880, 1],
            [0xD881, 0],
            [0xD881, 1],
        ]:
            ram[address] = set_bit(address, bit)


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


