import numpy as np

from deepred.polaris_env.gamestate import GameState
from deepred.polaris_env.pokemon_red.enums import RamLocation, DataStructDimension, Move
from deepred.polaris_env.pokemon_red.move_infos import MoveInfos


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
            MoveInfos[Move(gamestate._read(RamLocation.WHICH_POKEMON_LEARNED_MOVES + i))].actual_power(ptypes)
            for i in range(4)]

        new_move_power = MoveInfos[Move(gamestate._read(RamLocation.MOVE_TO_LEARN))].actual_power(ptypes)

        argmin_move = np.argmin(move_powers)
        if new_move_power > move_powers[argmin_move]:
            gamestate._ram[RamLocation.MENU_ITEM_ID] = argmin_move
            return True
        else:
            # do not replace, press B
            return False


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
