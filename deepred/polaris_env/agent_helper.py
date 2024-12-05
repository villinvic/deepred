from typing import Dict, List

import numpy as np

from deepred.polaris_env.game_patching import set_player_money, to_double
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.bag_item_info import BagItemsInfo, BagItemInfo
from deepred.polaris_env.pokemon_red.enums import RamLocation, DataStructDimension, Move, BagItem, Map, Pokemon
from deepred.polaris_env.pokemon_red.move_info import MovesInfo
from deepred.polaris_env.pokemon_red.pokemon_stats import PokemonStats, PokemonBaseStats


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

    def switch(
            self,
            gamestate: GameState
    ):
        """
        Skips the switch confirmation menu
        """
        if gamestate.menu_item_id != 0:
            gamestate._ram[RamLocation.MENU_ITEM_ID] = 0


    def learn_move(
            self,
            gamestate: GameState
    ) -> bool:
        """
        We automatically replace weaker moves
        :return Whether we should learn or not the move.
        """
        # TODO: we should move some of this to the gamestate class.
        party_pos = gamestate._read(RamLocation.WHICH_POKEMON)
        ptypes = [gamestate._read(RamLocation.PARTY_0_TYPE0 + (party_pos * DataStructDimension.POKEMON_STATS) + i)
                  for i in range(2)]
        move_powers = [
            MovesInfo[Move(gamestate._read(RamLocation.WHICH_POKEMON_LEARNED_MOVES + i))].actual_power(ptypes)
            for i in range(4)]

        new_move_power = MovesInfo[Move(gamestate._read(RamLocation.MOVE_TO_LEARN))].actual_power(ptypes)

        argmin_move = np.argmin(move_powers)
        if new_move_power > move_powers[argmin_move]:
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

        MAX_BAG_TYPES = 20
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

                # If our bag is full and we still have enough money:
                if bag_full() and money >= item_info.price:
                    sell_items_to_free_space(
                        current_bag,
                        money
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

           If the party is not full, just add the best pokemon.
           If the box is empty, do nothing.

           Returns if we interacted with the pc or not.

           TODO: as we will need the info at all time, we should keep in memory the box pokemons somehow.
                -> we need an observation telling us there is good pokemons in the box.
           """
        ram = gamestate._ram
        box_count = gamestate.box_pokemon_count

        box_pokemons = extract_box_pokemon_stats(gamestate)
        box_stat_sums = [box_pokemon.scale(50).sum() for box_pokemon in box_pokemons]
        best_box_pokemon = np.argmax(box_stat_sums)
        # If our party is not full, just move the best box pokemon into our party
        if gamestate.box_pokemon_count < 6:
            return send_box_pokemon_to_party(

                best_box_pokemon
            )

        party_stat_sums = []
        for i in range(gamestate.party_count):
            pass

        party_stat_sums = []
        worst_party_pokemon = np.argmin(party_stat_sums)

        if party_stat_sums[worst_party_pokemon] > box_stat_sums[best_box_pokemon]:
            # We found a pokemon that is worse that our team in the box, just release worst pokemons.
            return release_weak_pokemons(box_stat_sums, stat_sum_threshold=party_stat_sums[worst_party_pokemon])

        ram[RamLocation.BOX_POKEMON_COUNT] = new_boxcount


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


def extract_box_pokemon_stats(
        gamestate: GameState
) -> List[PokemonStats]:

    box_pokemons = []
    for i in range(gamestate.box_pokemon_count):
        offset = RamLocation.BOX_POKEMON_START + i * DataStructDimension.BOX_POKEMON_STATS
        species = gamestate._read(offset)
        level = gamestate._read(offset + 3)
        exp = gamestate._read_triple(offset + 14)
        hp_ev = gamestate._read_double(offset + 17)
        atk_ev = gamestate._read_double(offset + 19)
        def_ev = gamestate._read_double(offset + 21)
        spd_ev = gamestate._read_double(offset + 23)
        spc_ev = gamestate._read_double(offset + 25)
        atk_def_iv = gamestate._read(offset + 27)
        spd_spc_iv = gamestate._read(offset + 28)

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
        box_pokemons.append(pokemon_stats)

    return box_pokemons


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

    ram[slice(RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS,
    RamLocation.PARTY_START + (party_index + 1) * DataStructDimension.POKEMON_STATS, 1)] = ram[slice(
        RamLocation.BOX_POKEMON_START + box_index * DataStructDimension.BOX_POKEMON_STATS,
        RamLocation.BOX_POKEMON_START + (box_index + 1) * DataStructDimension.BOX_POKEMON_STATS, 1)
    ]

    scaled_stats = pokemon_stats.scale()

    b1, b2 = to_double(scaled_stats.hp)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35] = b2

    b1, b2 = to_double(scaled_stats.attack)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 2] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 2] = b2

    b1, b2 = to_double(scaled_stats.defense)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 4] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 4] = b2

    b1, b2 = to_double(scaled_stats.speed)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 6] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 6] = b2

    b1, b2 = to_double(scaled_stats.special)
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 34 + 8] = b1
    ram[RamLocation.PARTY_START + party_index * DataStructDimension.POKEMON_STATS + 35 + 8] = b2

    ram[slice(RamLocation.PARTY_NICKNAMES_START + party_index * DataStructDimension.POKEMON_NICKNAME,
              RamLocation.PARTY_NICKNAMES_START + (party_index + 1) * DataStructDimension.POKEMON_NICKNAME,
              1
              )] = ram[slice(
        RamLocation.BOX_NICKNAMES_START + box_index * DataStructDimension.POKEMON_NICKNAME,
        RamLocation.BOX_NICKNAMES_START + (box_index + 1) * DataStructDimension.POKEMON_NICKNAME, 1)]

    return delete_pokemon_box_at_index(
        ram,
        box_count,
        box_index
    )


def release_weak_pokemons(
        box_pokemon_stat_sums: List[int],
        threshold: int,
) -> int:
    """

    """


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

        ram[slice(RamLocation.BOX_POKEMON_START + i * DataStructDimension.BOX_POKEMON_STATS,
                  RamLocation.BOX_POKEMON_START + (i + 1) * DataStructDimension.BOX_POKEMON_STATS, 1)] = ram[slice(
            RamLocation.BOX_POKEMON_START + (i + 1) * DataStructDimension.BOX_POKEMON_STATS,
            RamLocation.BOX_POKEMON_START + (i + 2) * DataStructDimension.BOX_POKEMON_STATS, 1
        )]

        ram[slice(RamLocation.BOX_NICKNAMES_START + i * DataStructDimension.POKEMON_NICKNAME,
                  RamLocation.BOX_NICKNAMES_START + (i + 1) * DataStructDimension.POKEMON_NICKNAME, 1)] = ram[slice(
            RamLocation.BOX_NICKNAMES_START + (i + 1) * DataStructDimension.POKEMON_NICKNAME,
            RamLocation.BOX_NICKNAMES_START + (i + 2) * DataStructDimension.POKEMON_NICKNAME, 1
        )]

    new_boxcount = box_count - 1

    ram[RamLocation.BOX_POKEMON_SPECIES_START + new_boxcount] = 0xFF

    ram[slice(RamLocation.BOX_POKEMON_START + new_boxcount * DataStructDimension.BOX_POKEMON_STATS,
              RamLocation.BOX_POKEMON_START + (new_boxcount + 1) * DataStructDimension.BOX_POKEMON_STATS, 1)] = 0

    ram[slice(RamLocation.BOX_NICKNAMES_START + new_boxcount * DataStructDimension.POKEMON_NICKNAME,
              RamLocation.BOX_NICKNAMES_START + (new_boxcount + 1) * DataStructDimension.POKEMON_NICKNAME, 1)] = 0

    return new_boxcount


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

