from typing import Dict, List, Callable

import numpy as np
from pyboy.utils import WindowEvent

from deepred.polaris_env.game_patching import set_player_money, to_double
from deepred.polaris_env.gamestate import GameState, get_looking_at_coords
from deepred.polaris_env.pokemon_red.bag_item_info import BagItemsInfo, BagItemInfo
from deepred.polaris_env.pokemon_red.enums import RamLocation, DataStructDimension, BagItem, TileSet, FieldMove
from deepred.polaris_env.pokemon_red.pokemon_stats import PokemonStats


class AgentHelper:

    """

    - Auto switch
    - Auto learn better moves
    - routines for HMs, flute, other
    - manage pokemons in party, box
    - manage bag items
    - sell/buy items

    # TODO: they also block certain paths to accelerate learning.
    """

    def roll_party(
            self,
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

        ram = gamestate._ram

        # the pokemon at slot 0 will wrap around.
        slot_0 = {
            "species": ram[RamLocation.PARTY_0_ID],
            "stats": ram[slice(
            RamLocation.PARTY_START,
            RamLocation.PARTY_START + DataStructDimension.POKEMON_STATS,
            1
        )],
            "nickname": ram[slice(
            RamLocation.PARTY_NICKNAMES_START,
            RamLocation.PARTY_NICKNAMES_START + DataStructDimension.POKEMON_NICKNAME,
            1
        )]
        }

        for slot in range(gamestate.party_count - 1):
            # move each pokemon by one slot.
            # species
            ram[RamLocation.PARTY_0_ID + slot] = ram[RamLocation.PARTY_0_ID + (slot + 1)]
            ram[RamLocation.PARTY_0_ID + slot] = ram[RamLocation.PARTY_0_ID + (slot + 1)]

            for addr in range(RamLocation.PARTY_START + slot * DataStructDimension.POKEMON_STATS, RamLocation.PARTY_START + (slot + 1) * DataStructDimension.POKEMON_STATS):
                ram[addr] = ram[addr + DataStructDimension.POKEMON_STATS]

            for addr in range(RamLocation.PARTY_NICKNAMES_START + slot * DataStructDimension.POKEMON_NICKNAME, RamLocation.PARTY_NICKNAMES_START + (slot + 1) * DataStructDimension.POKEMON_NICKNAME):
                ram[addr] = ram[addr + DataStructDimension.POKEMON_NICKNAME]

        ram[RamLocation.PARTY_0_ID + (gamestate.party_count - 1)] = slot_0["species"]
        for i, addr in enumerate(range(RamLocation.PARTY_START + (gamestate.party_count - 1) * DataStructDimension.POKEMON_STATS,
                          RamLocation.PARTY_START + gamestate.party_count * DataStructDimension.POKEMON_STATS)):
            ram[addr] = slot_0["stats"][i]
        for i, addr in enumerate(range(RamLocation.PARTY_NICKNAMES_START + (gamestate.party_count - 1) * DataStructDimension.POKEMON_NICKNAME,
                          RamLocation.PARTY_NICKNAMES_START + gamestate.party_count * DataStructDimension.POKEMON_NICKNAME)):
            ram[addr] =  slot_0["nickname"][i]


    def switch(
            self,
            gamestate: GameState
    ):
        """
        Skips the switch confirmation menu
        """
        if gamestate.menu_item_id != 0:
            gamestate._ram[RamLocation.MENU_ITEM_ID] = 0


    def should_learn_move(
            self,
            gamestate: GameState
    ) -> bool:
        """
        We automatically replace weaker moves
        :return Whether we should learn or not the move.
        """
        learned_move_powers, new_move_power = gamestate.pokemon_move_learning_powers

        argmin_move = np.argmin(learned_move_powers)
        if new_move_power > learned_move_powers[argmin_move]:
            gamestate._ram[RamLocation.MENU_ITEM_ID] = argmin_move
            return True
        else:
            return False

    def shopping(
            self,
            gamestate: GameState
    ) -> bool:
        """
        For now, buy as much as possible
        Returns if we bought successfully or not.
        """

        MAX_BAG_TYPES = 20 - 1 # keep a free slot on top
        MAX_ITEM_COUNT = 10
        current_bag = gamestate.bag_items
        money = gamestate.player_money

        def bag_full():
            return len(current_bag) >= MAX_BAG_TYPES

        # Ensure bag starts within constraints
        money = sell_useless_items(current_bag, money)

        # Buy items in order of their priority
        ranked_mart_items = sorted(
            gamestate.mart_items,
            key=lambda i: -BagItemsInfo[i].priority,
        )

        for item in ranked_mart_items:
            item_info = BagItemsInfo[item]

            if bag_full() and money >= item_info.price:
                money = sell_items_to_free_space(
                    current_bag,
                    money
                )

            while (
                    money >= item_info.price
                    and
                    current_bag[item] < MAX_ITEM_COUNT
                    and not bag_full()
                    and item_info.priority > 0 # never buy useless items.
            ):
                money = buy_item(
                    current_bag,
                    money,
                    item,
                    item_info
                )


        ram = gamestate._ram
        inject_bag_to_ram(
            ram,
            current_bag
        )
        set_player_money(ram, money)

        # Always returns True
        return True

    def handle_full_bag(
            self,
            gamestate: GameState
    ):
        """
        TODO: Unused
        """
        bag = gamestate.bag_items
        sell_items_to_free_space(
            bag,
            toss=True
        )
        inject_bag_to_ram(gamestate._ram, bag)

    def manage_party(
            self,
            gamestate: GameState
    ) -> bool:
        """
        Moves the lowest total stats pokemon in the party with the highest stats pokemon in the box.

        The party should be full and there should be pokemons in the box if we call this function.
        """

        ram = gamestate._ram
        box_count = gamestate.box_pokemon_count

        party_stat_sums = gamestate.party_pokemon_stat_sums
        worst_party_pokemon_index = gamestate.worst_party_pokemon_index
        box_stat_sums = gamestate.box_pokemon_stat_sums
        best_box_pokemon_index = gamestate.best_box_pokemon_index

        if party_stat_sums[worst_party_pokemon_index] > box_stat_sums[best_box_pokemon_index]:
            box_count = release_weak_pokemons(
                gamestate._ram,
                box_count,
                box_stat_sums,
                threshold=party_stat_sums[worst_party_pokemon_index]
            )
        else:
            box_count = send_box_pokemon_to_party(
                ram,
                gamestate.box_pokemon_stats[best_box_pokemon_index],
                best_box_pokemon_index,
                worst_party_pokemon_index,
                box_count
            )

        ram[RamLocation.BOX_POKEMON_COUNT] = box_count

        return True

    def field_move(
            self,
            gamestate: GameState,
            step_function: Callable[[Callable, int, WindowEvent], None],
    ) -> bool :
        """
        This needs to interact with the emulator back and forth
        We pass the step function.
        We return True if any field move was used.
        No Flash/Strength as we do not need them.
        """
        return (
                try_use_flute(gamestate, step_function)
                or
                try_use_cut(gamestate, step_function)
                or
                try_use_surf(gamestate, step_function)
        )


def sell_useless_items(
        current_bag: Dict[BagItem, int],
        current_money: int = 0
) -> int:
    """
    sell all priority 0 items.
    """
    for item, quantity in list(current_bag.items()):
        item_info = BagItemsInfo[item]
        if item_info.priority == 0:
            current_money += quantity * item_info.sell_price
            del current_bag[item]
    return current_money


def sell_items_to_free_space(
        current_bag: Dict[BagItem, int],
        current_money: int = 0,
        toss: bool = False
) -> int:
    """
    We sell the worst item type in the bag.
    We can optionally pass toss=True if we are tossing (no money gained).
    """
    item_to_sell = sorted(
        list(current_bag.keys()),
        key=lambda i: BagItemsInfo[i].priority,
    )[0]

    item_info = BagItemsInfo[item_to_sell]
    if item_info.priority < 9:
        if not toss:
            current_money += item_info.sell_price * current_bag[item_to_sell]
        del current_bag[item_to_sell]

    return current_money


def buy_item(
        current_bag: Dict[BagItem, int],
        current_money: int,
        item: BagItem,
        item_info: BagItemInfo,
) -> int:
    current_bag[item] += 1
    current_money -= item_info.price
    return current_money

def bag_value(
        current_bag: Dict[BagItem, int]
) -> int:
    return sum([
        BagItemInfo(item).price * quantity for item, quantity in current_bag.items()
    ])

def inject_bag_to_ram(
        ram,
        bag: Dict[BagItem, int]
    ):
    """
    Writes the content of the bag into the ram.
    """
    for item_type_index, (item, quantity) in enumerate(bag.items()):
        offset = item_type_index * 2
        ram[RamLocation.BAG_ITEMS_START + offset] = item
        ram[RamLocation.BAG_ITEMS_START + offset + 1] = quantity

    # Fill the rest with dummy stuff
    for item_type_index in range(len(bag), 20):
        offset = item_type_index * 2
        ram[RamLocation.BAG_ITEMS_START + offset] = BagItem.NO_ITEM
        ram[RamLocation.BAG_ITEMS_START + offset + 1] = 0

    ram[RamLocation.BAG_COUNT] = len(bag)


def send_box_pokemon_to_party(
        ram,
        pokemon_stats: PokemonStats,
        box_index: int,
        party_index: int,
        box_count: int
) -> int:
    """
    Gets a pokemon from the box and replaces (and erases) one in the party.
    Returns new amount of pokemon in the box.
    """


    ram[RamLocation.PARTY_0_ID + party_index] = ram[RamLocation.BOX_POKEMON_SPECIES_START + box_index]

    for offset in range(DataStructDimension.BOX_POKEMON_STATS):
       ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + offset] =\
           ram[RamLocation.BOX_POKEMON_START + box_index * DataStructDimension.BOX_POKEMON_STATS + offset]
       if offset == 3:
           ram[RamLocation.PARTY_START + party_index * 44 + 33] = ram[RamLocation.BOX_POKEMON_START + box_index * 33 + 3]

    scaled_stats = pokemon_stats.scale()

    b1, b2 = to_double(scaled_stats.hp)
    print(b1, b2)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34] = b2

    b1, b2 = to_double(scaled_stats.attack)
    print(b1, b2)

    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 2] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 2] = b2

    b1, b2 = to_double(scaled_stats.defense)
    print(b1, b2)

    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 4] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 4] = b2

    b1, b2 = to_double(scaled_stats.speed)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 6] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 6] = b2

    b1, b2 = to_double(scaled_stats.special)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 8] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 8] = b2

    for offset in range(0, DataStructDimension.POKEMON_NICKNAME):
        ram[RamLocation.PARTY_NICKNAMES_START + party_index * DataStructDimension.POKEMON_NICKNAME + offset] = ram[
            RamLocation.BOX_NICKNAMES_START + box_index * DataStructDimension.POKEMON_NICKNAME + offset
        ]

    return delete_pokemon_box_at_index(
        ram,
        box_count,
        box_index
    )


def release_weak_pokemons(
        ram,
        box_count: int,
        box_pokemon_stat_sums: List[int],
        threshold: int,
) -> int:
    """
    releases all pokemons weaker than given threshold
    """
    for i in range(len(box_pokemon_stat_sums) - 1, 0, -1):
        if box_pokemon_stat_sums[i] <= threshold:
            box_count = delete_pokemon_box_at_index(
                ram,
                box_count,
                i
            )
    return box_count


def delete_pokemon_box_at_index(
        ram,
        box_count: int,
        index: int
):
    """
    Removes the pokemon at give index by shifting the pokemon indexed below up one spot.
    """

    for i in range(index, box_count - 1):
        ram[RamLocation.BOX_POKEMON_SPECIES_START + i] = ram[RamLocation.BOX_POKEMON_SPECIES_START + i + 1]

        for addr in range(RamLocation.BOX_POKEMON_START + i * DataStructDimension.BOX_POKEMON_STATS,
                          RamLocation.BOX_POKEMON_START + (i + 1) * DataStructDimension.BOX_POKEMON_STATS):
            ram[addr] = ram[addr + DataStructDimension.BOX_POKEMON_STATS]

        for addr in range(RamLocation.BOX_NICKNAMES_START + i * DataStructDimension.POKEMON_NICKNAME,
                          RamLocation.BOX_NICKNAMES_START + (i + 1) * DataStructDimension.POKEMON_NICKNAME):
            ram[addr] = ram[addr + DataStructDimension.POKEMON_NICKNAME]

    new_boxcount = box_count - 1

    ram[RamLocation.BOX_POKEMON_SPECIES_START + new_boxcount] = 0xFF

    for addr in range(RamLocation.BOX_POKEMON_START + new_boxcount * DataStructDimension.BOX_POKEMON_STATS,
                      RamLocation.BOX_POKEMON_START + (new_boxcount + 1) * DataStructDimension.BOX_POKEMON_STATS):
        ram[addr] = 0

    for addr in range(RamLocation.BOX_NICKNAMES_START + new_boxcount * DataStructDimension.POKEMON_NICKNAME,
                      RamLocation.BOX_NICKNAMES_START + (new_boxcount + 1) * DataStructDimension.POKEMON_NICKNAME):
        ram[addr] = 0

    return new_boxcount


def try_use_surf(
        gamestate: GameState,
        step_function: Callable[[Callable, int, WindowEvent], None]
) -> bool:
    """
    Attempts at using surf, we need to have the ability to surf, to be standing on the ground
    and to be looking at the right tile.
    """
    #if not (gamestate.can_use_surf and gamestate.tileset_id in WaterTilesets):
    #    return False
    if gamestate.walk_bike_surf_state == 2:
        return False

    x, y = get_looking_at_coords(gamestate)
    looking_at_tile = gamestate.bottom_left_screen_tiles[y, x]
    stand_on_tile = gamestate.bottom_left_screen_tiles[4, 4]

    if (
        gamestate.tileset_id == TileSet.TILESET_17
        and
        looking_at_tile == 0x14
        and
        stand_on_tile == 0x05
    ):
        return False
    elif (
        gamestate.tileset_id == TileSet.FOREST
        and
        looking_at_tile in [0x14, 0x48]
        and stand_on_tile == 0x2E
    ):
        return False
    elif (
        gamestate.tileset_id in [TileSet.TILESET_13, TileSet.VERMILION_PORT]
        and looking_at_tile != 0x14
    ):
        return False
    elif looking_at_tile != 0x14:
        return False

    # Prevent bot from using surf in some places ?

    use_field_move(FieldMove.SURF, gamestate, step_function)

    return True


def try_use_cut(
        gamestate: GameState,
        step_function: Callable[[Callable, int, WindowEvent], None]
) -> bool:
    """
    Attempts at using cut, we need to have the ability to cut, and to be looking at the right tile.
    """

    if not (gamestate.can_use_cut and gamestate.tileset_id in (TileSet.OVERWORLD, TileSet.GYM)):
       return False

    trees = gamestate.feature_maps[1]
    x, y = get_looking_at_coords(gamestate)

    if trees[y, x] != 255:
        return False

    use_field_move(FieldMove.CUT, gamestate, step_function)

    return True


def try_use_flute(
        gamestate: GameState,
        step_function: Callable[[Callable, int, WindowEvent], None]
) -> bool :
    """
    Attempts at using the flute, we need to have the ability to use it, and to be looking at a snorlax.
    """

    if not (gamestate.can_use_flute and gamestate.tileset_id == TileSet.OVERWORLD):
        return False

    x, y = get_looking_at_coords(gamestate)
    if gamestate.sprite_map[y, x] != 32: # Snorlax
        return False

    flute_bag_idx = gamestate.ordered_bag_items.index(BagItem.POKE_FLUTE)
    def ram_modifier(_ram):
        _ram[RamLocation.BATTLE_SAVED_ITEM] = 2
        _ram[RamLocation.BAG_SAVED_ITEM] = 0
        _ram[RamLocation.SCROLL_OFFSET_VALUE] = flute_bag_idx

    step_function(ram_modifier, 18, WindowEvent.PRESS_BUTTON_START)
    for _ in range(10):
        step_function(ram_modifier, 18, WindowEvent.PRESS_BUTTON_A)

    return True


def use_field_move(
    field_move: FieldMove,
    gamestate: GameState,
    step_function: Callable[[Callable, int, WindowEvent], None]
):
    ram = gamestate._ram # ram is shared across gamestates.

    ram[RamLocation.BATTLE_SAVED_ITEM] = 1  # 1 is pokemon
    def ram_modifier(_ram):
        _ram[RamLocation.FIELD_MOVE] = field_move
        _ram[RamLocation.WHICH_POKEMON] = 0
        _ram[RamLocation.MAX_MENU_ITEM] = 3

    step_function(ram_modifier, 16, WindowEvent.PRESS_BUTTON_START)
    step_function(ram_modifier, 16, WindowEvent.PRESS_BUTTON_A)

    for _ in range(3):
        step_function(ram_modifier, 16, WindowEvent.PRESS_BUTTON_A)
    step_function(ram_modifier, 24*3, WindowEvent.PASS)


