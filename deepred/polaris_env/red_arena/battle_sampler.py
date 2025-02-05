from dataclasses import dataclass
from enum import IntEnum
from typing import List, Dict, NamedTuple, Tuple

import tree

from deepred.polaris_env.agent_helper import inject_bag_to_ram
from deepred.polaris_env.battle_parser import CHARMAP
from deepred.polaris_env.game_patching import to_double, to_triple
from deepred.polaris_env.pokemon_red.bag_item_info import BagItemsInfo, key_item
from deepred.polaris_env.pokemon_red.enums import Pokemon, Move, RamLocation, DataStructDimension, BagItem, BattleType
from deepred.polaris_env.pokemon_red.move_info import MovesInfo
from deepred.polaris_env.pokemon_red.pokemon_stats import PokemonStats, PokemonsBaseStats, PokemonBaseStats

import numpy as np

from deepred.polaris_env.red_arena.pokemon_data import PokemonDatas


class UsablePokemon(IntEnum):
    RHYDON = 0x1
    KANGASKHAN = 0x2
    NIDORANM = 0x3
    CLEFAIRY = 0x4
    SPEAROW = 0x5
    VOLTORB = 0x6
    NIDOKING = 0x7
    SLOWBRO = 0x8
    IVYSAUR = 0x9
    EXEGGUTOR = 0xA
    LICKITUNG = 0xB
    EXEGGCUTE = 0xC
    GRIMER = 0xD
    GENGAR = 0xE
    NIDORANF = 0xF
    NIDOQUEEN = 0x10
    CUBONE = 0x11
    RHYHORN = 0x12
    LAPRAS = 0x13
    ARCANINE = 0x14
    MEW = 0x15
    GYARADOS = 0x16
    SHELLDER = 0x17
    TENTACOOL = 0x18
    GASTLY = 0x19
    SCYTHER = 0x1A
    STARYU = 0x1B
    BLASTOISE = 0x1C
    PINSIR = 0x1D
    TANGELA = 0x1E
    GROWLITHE = 0x21
    ONIX = 0x22
    FEAROW = 0x23
    PIDGEY = 0x24
    SLOWPOKE = 0x25
    KADABRA = 0x26
    GRAVELER = 0x27
    CHANSEY = 0x28
    MACHOKE = 0x29
    MR_MIME = 0x2A
    HITMONLEE = 0x2B
    HITMONCHAN = 0x2C
    ARBOK = 0x2D
    PARASECT = 0x2E
    PSYDUCK = 0x2F
    DROWZEE = 0x30
    GOLEM = 0x31
    MAGMAR = 0x33
    ELECTABUZZ = 0x35
    MAGNETON = 0x36
    KOFFING = 0x37
    MANKEY = 0x39
    SEEL = 0x3A
    DIGLETT = 0x3B
    TAUROS = 0x3C
    FARFETCHD = 0x40
    VENONAT = 0x41
    DRAGONITE = 0x42
    DODUO = 0x46
    POLIWAG = 0x47
    JYNX = 0x48
    MOLTRES = 0x49
    ARTICUNO = 0x4A
    ZAPDOS = 0x4B
    DITTO = 0x4C
    MEOWTH = 0x4D
    KRABBY = 0x4E
    VULPIX = 0x52
    NINETALES = 0x53
    PIKACHU = 0x54
    RAICHU = 0x55
    DRATINI = 0x58
    DRAGONAIR = 0x59
    KABUTO = 0x5A
    KABUTOPS = 0x5B
    HORSEA = 0x5C
    SEADRA = 0x5D
    SANDSHREW = 0x60
    SANDSLASH = 0x61
    OMANYTE = 0x62
    OMASTAR = 0x63
    JIGGLYPUFF = 0x64
    WIGGLYTUFF = 0x65
    EEVEE = 0x66
    FLAREON = 0x67
    JOLTEON = 0x68
    VAPOREON = 0x69
    MACHOP = 0x6A
    ZUBAT = 0x6B
    EKANS = 0x6C
    PARAS = 0x6D
    POLIWHIRL = 0x6E
    POLIWRATH = 0x6F
    WEEDLE = 0x70
    KAKUNA = 0x71
    BEEDRILL = 0x72
    DODRIO = 0x74
    PRIMEAPE = 0x75
    DUGTRIO = 0x76
    VENOMOTH = 0x77
    DEWGONG = 0x78
    CATERPIE = 0x7B
    METAPOD = 0x7C
    BUTTERFREE = 0x7D
    MACHAMP = 0x7E
    GOLDUCK = 0x80
    HYPNO = 0x81
    GOLBAT = 0x82
    MEWTWO = 0x83
    SNORLAX = 0x84
    MAGIKARP = 0x85
    MUK = 0x88
    KINGLER = 0x8A
    CLOYSTER = 0x8B
    ELECTRODE = 0x8D
    CLEFABLE = 0x8E
    WEEZING = 0x8F
    PERSIAN = 0x90
    MAROWAK = 0x91
    HAUNTER = 0x93
    ABRA = 0x94
    ALAKAZAM = 0x95
    PIDGEOTTO = 0x96
    PIDGEOT = 0x97
    STARMIE = 0x98
    BULBASAUR = 0x99
    VENUSAUR = 0x9A
    TENTACRUEL = 0x9B
    GOLDEEN = 0x9D
    SEAKING = 0x9E
    PONYTA = 0xA3
    RAPIDASH = 0xA4
    RATTATA = 0xA5
    RATICATE = 0xA6
    NIDORINO = 0xA7
    NIDORINA = 0xA8
    GEODUDE = 0xA9
    PORYGON = 0xAA
    AERODACTYL = 0xAB
    MAGNEMITE = 0xAD
    CHARMANDER = 0xB0
    SQUIRTLE = 0xB1
    CHARMELEON = 0xB2
    WARTORTLE = 0xB3
    CHARIZARD = 0xB4
    ODDISH = 0xB9
    GLOOM = 0xBA
    VILEPLUME = 0xBB
    BELLSPROUT = 0xBC
    WEEPINBELL = 0xBD
    VICTREEBEL = 0xBE


class SampledPokemon(NamedTuple):
    stats: PokemonStats
    moves: List[Move]

    def inject_at(
            self,
            ram,
            index: int,
            is_opponent: bool
    ):
        """
        :param ram: ram to modify
        :param index: index of the pokemon in the party (0 for leading / wild pokemon)
        :param is_opponent: if this is about the opponent / wild pokemon
        """

        if is_opponent:
            # look into https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Battle
            #           https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Opponent_Trainer%E2%80%99s_Pok%C3%A9mon
            # look into how the game handles wild encounters, and trainer battle initialisation: https://github.com/pret/pokered
            # also into opponent_sent_out_pokemon_stats.py (line 638)

            if index == 0:
                ram[RamLocation.ENEMY_POKEMON_SPECIES] = self.stats.pokemon
                ram[RamLocation.ENEMY_POKEMON_ID] = self.stats.pokemon
                ram[RamLocation.ENEMY_LEVEL] = self.stats.level
                ram[RamLocation.ENEMY_LEVEL + 14] = self.stats.level

                hp_iv = (
                        ((self.stats.ivs.attack & 1) << 0) |
                        ((self.stats.ivs.attack & 1) << 1) |
                        ((self.stats.ivs.speed & 1) << 2) |
                        ((self.stats.ivs.speed & 1) << 3)
                )

                ram[0xCFF1] = self.stats.ivs.attack
                ram[0xCFF2] = self.stats.ivs.speed

                self.stats.ivs = PokemonBaseStats(hp=hp_iv, attack=self.stats.ivs.attack, defense=self.stats.ivs.attack, speed=self.stats.ivs.speed, special=self.stats.ivs.speed)

                scaled_stats = self.stats.scale()
                # HP
                a, b = to_double(scaled_stats.hp)
                ram[RamLocation.ENEMY_POKEMON_MAX_HP] = b
                ram[RamLocation.ENEMY_POKEMON_MAX_HP + 1] = a

                ram[RamLocation.ENEMY_POKEMON_HP] = b
                ram[RamLocation.ENEMY_POKEMON_HP + 1] = a

                # Status
                ram[RamLocation.ENEMY_POKEMON_STATUS] = 0

                pokemon_data = PokemonDatas[self.stats.pokemon]

                # Types
                ram[RamLocation.ENEMY_POKEMON_TYPE0] = pokemon_data.types[0].unfix()
                ram[RamLocation.ENEMY_POKEMON_TYPE1] = pokemon_data.types[1].unfix()

                #  Moves
                for i, move in enumerate(self.moves + [Move.NO_MOVE] * (4 - len(self.moves))):
                    ram[RamLocation.ENEMY_POKEMON_MOVE0 + i] = move
                    ram[RamLocation.ENEMY_POKEMON_MOVE0_PP + i] = MovesInfo[move].pp

                # Catch Rate
                ram[RamLocation.ENEMY_POKEMON_CATCH_RATE] = pokemon_data.catch_rate

                # Attack
                a, b = to_double(scaled_stats.attack)
                ram[RamLocation.ENEMY_POKEMON_ATTACK] = b
                ram[RamLocation.ENEMY_POKEMON_ATTACK + 1] = a

                # Defense
                a, b = to_double(scaled_stats.defense)
                ram[RamLocation.ENEMY_POKEMON_DEFENSE] = b
                ram[RamLocation.ENEMY_POKEMON_DEFENSE + 1] = a

                # Speed
                a, b = to_double(scaled_stats.speed)
                ram[RamLocation.ENEMY_POKEMON_SPEED] = b
                ram[RamLocation.ENEMY_POKEMON_SPEED + 1] = a

                # Special
                a, b = to_double(scaled_stats.special)
                ram[RamLocation.ENEMY_POKEMON_SPECIAL] = b
                ram[RamLocation.ENEMY_POKEMON_SPECIAL + 1] = a

                # Experience
                e1, e2, e3 = to_triple(self.stats.exp)
                ram[RamLocation.ENEMY_POKEMON_EXPERIENCE + index * DataStructDimension.POKEMON_STATS] = e3


            if index >= 0:
                ram[RamLocation.OPPONENT_POKEMON_0_ID + index] = self.stats.pokemon
                ram[RamLocation.OPPONENT_POKEMON_0_SPECIES + index * DataStructDimension.POKEMON_STATS] = self.stats.pokemon
                ram[RamLocation.OPPONENT_POKEMON_0_LEVEL + index * DataStructDimension.POKEMON_STATS] = self.stats.level
                ram[RamLocation.OPPONENT_POKEMON_0_STATUS + index * DataStructDimension.POKEMON_STATS] = 0

                pokemon_data = PokemonDatas[self.stats.pokemon]

                # Types
                ram[RamLocation.OPPONENT_POKEMON_0_TYPE0 + index * DataStructDimension.POKEMON_STATS] = pokemon_data.types[0].unfix()
                ram[RamLocation.OPPONENT_POKEMON_0_TYPE1 + index * DataStructDimension.POKEMON_STATS] = pokemon_data.types[1].unfix()

                # Moves
                for i, move in enumerate(self.moves + [Move.NO_MOVE] * (4 - len(self.moves))):
                    ram[RamLocation.OPPONENT_POKEMON_0_MOVE0 + i + index * DataStructDimension.POKEMON_STATS] = move
                    ram[RamLocation.OPPONENT_POKEMON_0_MOVE0_PP + i + index * DataStructDimension.POKEMON_STATS] = MovesInfo[move].pp

                scaled_stats = self.stats.scale()

                # HP
                a, b = to_double(scaled_stats.hp) # There is modification of maxhp after ~10 frames after the start of the fight
                ram[RamLocation.OPPONENT_POKEMON_0_MAX_HP + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_MAX_HP + 1 + index * DataStructDimension.POKEMON_STATS] = a

                ram[RamLocation.OPPONENT_POKEMON_0_HP + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_HP + 1 + index * DataStructDimension.POKEMON_STATS] = a

                # Attack
                a, b = to_double(scaled_stats.attack)
                ram[RamLocation.OPPONENT_POKEMON_0_ATTACK + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_ATTACK + 1 + index * DataStructDimension.POKEMON_STATS] = a

                # Defense

                a, b = to_double(scaled_stats.defense)
                ram[RamLocation.OPPONENT_POKEMON_0_DEFENSE + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_DEFENSE + 1 + index * DataStructDimension.POKEMON_STATS] = a

                # Speed
                a, b = to_double(scaled_stats.speed)
                ram[RamLocation.OPPONENT_POKEMON_0_SPEED + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_SPEED + 1 + index * DataStructDimension.POKEMON_STATS] = a

                # Special
                a, b = to_double(scaled_stats.special)
                ram[RamLocation.OPPONENT_POKEMON_0_SPECIAL + index * DataStructDimension.POKEMON_STATS] = b
                ram[RamLocation.OPPONENT_POKEMON_0_SPECIAL + 1 + index * DataStructDimension.POKEMON_STATS] = a

                # Experience
                e1, e2, e3 = to_triple(self.stats.exp)
                ram[RamLocation.OPPONENT_POKEMON_0_EXPERIENCE + index * DataStructDimension.POKEMON_STATS] = e3
                ram[RamLocation.OPPONENT_POKEMON_0_EXPERIENCE + index * DataStructDimension.POKEMON_STATS + 1] = e2
                ram[RamLocation.OPPONENT_POKEMON_0_EXPERIENCE + index * DataStructDimension.POKEMON_STATS + 2] = e1
        else:
            poke_name = self.stats.pokemon.name
            for offset in range(0, DataStructDimension.POKEMON_NICKNAME):
                if offset < len(poke_name):
                    c = CHARMAP.get(poke_name[offset], CHARMAP["<NULL>"])
                else:
                    c = CHARMAP["<NULL>"]
                ram[RamLocation.PARTY_NICKNAMES_START + index * DataStructDimension.POKEMON_NICKNAME + offset] = c

            # we also have to set the species here for our party
            ram[RamLocation.PARTY_0_ID + index] = self.stats.pokemon
            ram[RamLocation.PARTY_0_SPECIES + index * DataStructDimension.POKEMON_STATS] = self.stats.pokemon

            # Status
            ram[RamLocation.PARTY_0_STATUS + index * DataStructDimension.POKEMON_STATS] = 0

            scaled_stats = self.stats.scale()
            pokemon_data = PokemonDatas[self.stats.pokemon]

            # Types
            ram[RamLocation.OPPONENT_POKEMON_0_TYPE0 + index * DataStructDimension.POKEMON_STATS] = pokemon_data.types[0].unfix()
            ram[RamLocation.OPPONENT_POKEMON_0_TYPE1 + index * DataStructDimension.POKEMON_STATS] = pokemon_data.types[1].unfix()

            # level
            ram[RamLocation.PARTY_0_LEVEL + index * DataStructDimension.POKEMON_STATS] = self.stats.level
            # 'level'
            ram[RamLocation.PARTY_0_FAKE_LEVEL + index * DataStructDimension.POKEMON_STATS] = self.stats.level

            # catch rate
            ram[RamLocation.PARTY_0_CATCH_RATE + index * DataStructDimension.POKEMON_STATS] = pokemon_data.catch_rate

            # exp
            e1, e2, e3 = to_triple(self.stats.exp)
            ram[RamLocation.PARTY_0_EXP + index * DataStructDimension.POKEMON_STATS] = e3
            ram[RamLocation.PARTY_0_EXP + index * DataStructDimension.POKEMON_STATS + 1] = e2
            ram[RamLocation.PARTY_0_EXP + index * DataStructDimension.POKEMON_STATS + 2] = e1

            # moves and pps
            for i, move in enumerate(self.moves + [Move.NO_MOVE] * (4 - len(self.moves))):
                ram[RamLocation.PARTY_0_MOVE0 + i + index * DataStructDimension.POKEMON_STATS] = move
                ram[RamLocation.PARTY_0_MOVE0_PP + i + index * DataStructDimension.POKEMON_STATS] = MovesInfo[move].pp

            # trainer ID (use the ID of pokemon 1)
            ram[RamLocation.PARTY_0_TRAINER_ID + index * DataStructDimension.POKEMON_STATS] = ram[RamLocation.PARTY_0_TRAINER_ID]
            ram[RamLocation.PARTY_0_TRAINER_ID + index * DataStructDimension.POKEMON_STATS + 1] = ram[RamLocation.PARTY_0_TRAINER_ID + 1]

            b1, b2 = to_double(scaled_stats.hp)

            # max hp
            ram[RamLocation.PARTY_0_MAXHP + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_MAXHP + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # current hp
            ram[RamLocation.PARTY_0_HP + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_HP + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Attack
            b1, b2 = to_double(scaled_stats.attack)
            ram[RamLocation.PARTY_0_ATTACK + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_ATTACK + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Defense
            b1, b2 = to_double(scaled_stats.defense)
            ram[RamLocation.PARTY_0_DEFENSE + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_DEFENSE + index * DataStructDimension.POKEMON_STATS+ 1] = b1

            # Speed
            b1, b2 = to_double(scaled_stats.speed)
            ram[RamLocation.PARTY_0_SPEED + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_SPEED + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Special
            b1, b2 = to_double(scaled_stats.special)
            ram[RamLocation.PARTY_0_SPECIAL + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_SPECIAL + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # HP Ev
            b1, b2 = to_double(self.stats.evs.hp)
            ram[RamLocation.PARTY_0_HP_EV + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_HP_EV + index * DataStructDimension.POKEMON_STATS + 1] = b1

            #Attack EV
            b1, b2 = to_double(self.stats.evs.attack)
            ram[RamLocation.PARTY_0_ATTACK_EV + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_ATTACK_EV + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Defense EV
            b1, b2 = to_double(self.stats.evs.defense)
            ram[RamLocation.PARTY_0_DEFENSE_EV + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_DEFENSE_EV + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Speed EV
            b1, b2 = to_double(self.stats.evs.speed)
            ram[RamLocation.PARTY_0_SPEED_EV + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_SPEED_EV + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # Special EV
            b1, b2 = to_double(self.stats.evs.special)
            ram[RamLocation.PARTY_0_SPECIAL_EV + index * DataStructDimension.POKEMON_STATS] = b2
            ram[RamLocation.PARTY_0_SPECIAL_EV + index * DataStructDimension.POKEMON_STATS + 1] = b1

            # IVs
            hp_iv = (
                    ((self.stats.ivs.attack & 1) << 0) |
                    ((self.stats.ivs.attack & 1) << 1) |
                    ((self.stats.ivs.speed & 1) << 2) |
                    ((self.stats.ivs.speed & 1) << 3)
            )
            self.stats.ivs = PokemonBaseStats(hp=hp_iv, attack=self.stats.ivs.attack, defense=self.stats.ivs.attack,
                                              speed=self.stats.ivs.speed, special=self.stats.ivs.speed)

            ram[RamLocation.PARTY_0_ATTACK_DEFENSE_IV] = self.stats.ivs.attack
            ram[RamLocation.PARTY_0_SPEED_SPECIAL_IV] = self.stats.ivs.speed


class SampledBattle(NamedTuple):
    savestate: str
    teams: Tuple[List[SampledPokemon], List[SampledPokemon]]
    bag: Dict[BagItem, int]

    def inject_to_ram(
            self,
            ram
    ):
        inject_bag_to_ram(ram, self.bag)
        inject_team_to_ram(ram, self.teams[0], is_opponent=False)
        inject_team_to_ram(ram, self.teams[1], is_opponent=True)


class BattleSampler:
    """
    Class to use for sampling battles

    Goal:
    -> samples plausible battles that can be encountered in game
    """

    def __init__(
            self,
            wild_battle_savestate: str,
            trainer_battle_savestate: str,
            level_mean_bounds: Tuple[int, int] = (5, 60),
            party_level_std_max: int = 10,
            opponent_level_std_max: int = 3,
            mean_item_quantity: float = 5.,
            mean_num_item_types: float = 3.,
            wild_battle_chance: float = 0.5,
    ):
        """
        :param wild_battle_savestate: Savestate for wild battles
        :param trainer_battle_savestate: Savestate for trainer battles
        :param level_mean_bounds: min and max values for mean party levels
        :param party_level_std_max: max standard deviation for party levels
        :param opponent_level_std_max: max standard deviation for opponent levels
        :param mean_item_quantity: mean quantity of items in the bag
        :param mean_num_item_types: mean number of item types in the bag
        :param wild_battle_chance: probability of having a wild battle
        """

        self.wild_battle_savestate = wild_battle_savestate
        self.trainer_battle_savestate = trainer_battle_savestate
        self.party_level_std_max = party_level_std_max
        self.opponent_level_std_max = opponent_level_std_max
        self.level_mean_bounds = level_mean_bounds
        self.mean_item_quantity = mean_item_quantity
        self.mean_num_item_types = mean_num_item_types
        self.wild_battle_chance = wild_battle_chance

    def __call__(
            self,
    ) -> SampledBattle:
        """
        injects the type of battle, the team and opponent's team to the ram.
        """
        is_wild = np.random.random() < 1#self.wild_battle_chance
        path = self.wild_battle_savestate if is_wild else self.trainer_battle_savestate
        bag = self.sample_bag()
        global_level_mean = np.random.randint(*self.level_mean_bounds)
        global_level_std = np.random.uniform(0, global_level_mean / 5)

        teams = (
            self.sample_team(
                level_mean=int(np.clip(np.random.normal(global_level_mean, global_level_std), *self.level_mean_bounds)),
                level_std=np.random.uniform(0, self.party_level_std_max),
                is_opponent=False,
                is_wild=False)
            ,
            self.sample_team(
                level_mean=int(np.clip(np.random.normal(global_level_mean, global_level_std), *self.level_mean_bounds)),
                level_std=np.random.uniform(0, self.opponent_level_std_max),
                is_opponent=True,
                is_wild=is_wild)
        )

        return SampledBattle(
            savestate=path,
            bag=bag,
            teams=tuple(teams)
        )

    def sample_team(
            self,
            level_mean: float,
            level_std: float,
            is_opponent: bool,
            is_wild: bool
    ) -> List[SampledPokemon]:

        """
        Randomly picks a team
        """
        if is_wild:
            team_size = 1
        else:
            team_size = np.random.randint(1, 7)

        return [
            self.sample_pokemon(
                level_mean,
                level_std,
                is_opponent
            )
            for _ in range(team_size)
        ]

    def sample_bag(
            self,
    ) -> Dict[BagItem, int]:

        num_item_types = round(np.clip(np.random.exponential(self.mean_num_item_types), 0, 20))
        if num_item_types == 0:
            return {}

        num_items = round(np.maximum(np.random.exponential(self.mean_item_quantity) * num_item_types, 1))

        samplable_items = [
            item for item in BagItem if (not ("FLOOR" in item.name or "BADGE" in item.name or "ITEM" in item.name) and
                                         item != BagItem.NO_ITEM)
        ]

        probs = np.exp([
            BagItemsInfo[item].priority / 10 if BagItemsInfo[item].priority != key_item else 1 for item in
            samplable_items
        ])
        probs /= probs.sum()

        item_types = np.random.choice(samplable_items, size=num_item_types, p=probs, replace=False)

        probs = np.exp([
            BagItemsInfo[item].priority / 10 if BagItemsInfo[item].priority != key_item else 1 for item in item_types
        ])
        probs /= probs.sum()

        bag = {}
        for _ in range(num_items):
            sampled = int(np.random.choice(item_types, p=probs))
            if not sampled in bag:
                bag[sampled] = 1
            elif BagItemsInfo[sampled].priority != key_item:
                bag[sampled] += 1

        return bag

    def sample_pokemon(
            self,
            level_mean: float,
            level_std: float,
            is_opponent: bool,
    ):
        # random moves, etc...

        pokemon = Pokemon(int(np.random.choice(UsablePokemon)))
        pokemon_data = PokemonDatas[pokemon]
        level = int(np.clip(np.random.normal(level_mean, level_std), 3, 100))

        if level == 100:
            exp = pokemon_data.get_exp(level)
        else:
            exp_min = pokemon_data.get_exp(level)
            exp_max = pokemon_data.get_exp(np.minimum(level + 1, 100))
            exp = int(np.random.uniform(exp_min, exp_max))

        evs = PokemonBaseStats()
        if not is_opponent:
            evs = tree.map_structure(
                lambda v: np.random.randint(0, 25600),  # max vitamins
                evs
            )

        ivs = tree.map_structure(
            lambda v: np.random.randint(0, 16),  # max iv
            evs
        )

        pokemon_stats = PokemonStats(
            pokemon=pokemon,
            level=level,
            exp=exp,
            evs=evs,
            ivs=ivs
        )

        # for now, assume party is fully healed ?
        # full pps, full hp, no status
        max_moves = np.minimum(4, len(pokemon_data.move_set))
        p = np.array([0.5, 0.4, 0.07, 0.03][:max_moves])
        p /= p.sum()
        if level < 12:
            num_moves = np.random.choice(max_moves, p=p) + 1
        else:
            num_moves = max_moves

        return SampledPokemon(stats=pokemon_stats,
                              moves=list(np.random.choice(list(pokemon_data.move_set), size=num_moves, replace=False)))


def inject_team_to_ram(
        ram,
        team: List[SampledPokemon],
        is_opponent: bool,
):
    if is_opponent:
        party_count_addr = RamLocation.OPPONENT_PARTY_COUNT
    else:
        party_count_addr = RamLocation.PARTY_COUNT
        ram[RamLocation.PARTY_0_ID + len(team)] = 255


    ram[party_count_addr] = len(team)
    for i, pokemon in enumerate(team):
        pokemon.inject_at(ram, i, is_opponent=is_opponent)
