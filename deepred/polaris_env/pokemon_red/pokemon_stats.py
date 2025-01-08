from typing import NamedTuple
import numpy as np

from deepred.polaris_env.pokemon_red.enums import Pokemon


def compute_stat(
        base_stat,
        level,
        ev,
        iv,
        is_hp=False
) -> int:
    """
    returns the stat amount for the specified level
    hp computation is different.
    """
    if is_hp:
        return int(((base_stat + iv) * 2 + np.sqrt(ev) / 4) * level / 100) + level + 10
    else:
        return int(((base_stat + iv) * 2 + np.sqrt(ev) / 4) * level / 100) + 5

def rescale_as(
        stat,
        original_level,
        target_level,
        is_hp=False
):
    """
    Rescales the stats
    """
    if is_hp:
        return int((stat - 10 - original_level) * target_level / original_level + target_level + 10)
    else:
        return int((stat - 5) * target_level / original_level + 5)


class PokemonStats:
    def __init__(
            self,
            pokemon: Pokemon,
            level: int,
            exp: int,
            evs: "PokemonBaseStats",
            ivs: "PokemonBaseStats",
    ):
        self.pokemon = pokemon
        self.base_stats = PokemonsBaseStats[pokemon]
        self.level = level
        self.exp = exp
        self.evs = evs
        self.ivs = ivs

    def scale(
            self,
            level: int | None = None
    )-> "PokemonBaseStats":
        """
        Returns the stats of the pokemon, scaled to the level (can specify a custom level), ev, and iv.
        """

        level = self.level if level is None else level
        return PokemonBaseStats(
            hp=compute_stat(self.base_stats.hp, level, self.evs.hp, self.ivs.hp, is_hp=True),
            attack=compute_stat(self.base_stats.attack, level, self.evs.attack, self.ivs.attack),
            defense=compute_stat(self.base_stats.defense, level, self.evs.defense, self.ivs.defense),
            speed=compute_stat(self.base_stats.speed, level, self.evs.speed, self.ivs.speed),
            special=compute_stat(self.base_stats.special, level, self.evs.special, self.ivs.special),
        )

    def __repr__(self):
        return f"PokemonStats(pokemon={self.pokemon}, level={self.level}, exp={self.exp}, evs={self.evs}, ivs={self.ivs})"


class PokemonBaseStats(NamedTuple):
    hp: int = 0
    attack: int = 0
    defense: int = 0
    speed: int = 0
    special: int = 0

    def rescale_as(
            self,
            original_level: int,
            target_level: int,
    ) -> "PokemonBaseStats":
        """
        Simulates the stats for another given level (the stats should be the true stats, not base).
        """
        return PokemonBaseStats(
            hp=rescale_as(self.hp, original_level, target_level, is_hp=True),
            attack=rescale_as(self.attack, original_level, target_level),
            defense=rescale_as(self.defense, original_level, target_level),
            speed=rescale_as(self.speed, original_level, target_level),
            special=rescale_as(self.special, original_level, target_level)
        )

    def sum(self):
        return (
                self.speed * 1.3 # those stats are very strong in gen 1
            +   self.special * 1.3

            +   self.hp
            +   self.defense
            +   self.attack
        )


PokemonsBaseStats = {
    Pokemon.ABRA: PokemonBaseStats(hp=25, attack=20, defense=15, speed=90, special=105),
    Pokemon.AERODACTYL: PokemonBaseStats(hp=80, attack=105, defense=65, speed=130, special=60),
    Pokemon.ALAKAZAM: PokemonBaseStats(hp=55, attack=50, defense=45, speed=120, special=135),
    Pokemon.ARBOK: PokemonBaseStats(hp=60, attack=85, defense=69, speed=80, special=65),
    Pokemon.ARCANINE: PokemonBaseStats(hp=90, attack=110, defense=80, speed=95, special=80),
    Pokemon.ARTICUNO: PokemonBaseStats(hp=90, attack=85, defense=100, speed=85, special=125),
    Pokemon.BEEDRILL: PokemonBaseStats(hp=65, attack=80, defense=40, speed=75, special=45),
    Pokemon.BELLSPROUT: PokemonBaseStats(hp=50, attack=75, defense=35, speed=40, special=70),
    Pokemon.BLASTOISE: PokemonBaseStats(hp=79, attack=83, defense=100, speed=78, special=85),
    Pokemon.BULBASAUR: PokemonBaseStats(hp=45, attack=49, defense=49, speed=45, special=65),
    Pokemon.BUTTERFREE: PokemonBaseStats(hp=60, attack=45, defense=50, speed=70, special=80),
    Pokemon.CATERPIE: PokemonBaseStats(hp=45, attack=30, defense=35, speed=45, special=20),
    Pokemon.CHANSEY: PokemonBaseStats(hp=250, attack=5, defense=5, speed=50, special=105),
    Pokemon.CHARIZARD: PokemonBaseStats(hp=78, attack=84, defense=78, speed=100, special=85),
    Pokemon.CHARMANDER: PokemonBaseStats(hp=39, attack=52, defense=43, speed=65, special=50),
    Pokemon.CHARMELEON: PokemonBaseStats(hp=58, attack=64, defense=58, speed=80, special=65),
    Pokemon.CLEFABLE: PokemonBaseStats(hp=95, attack=70, defense=73, speed=60, special=85),
    Pokemon.CLEFAIRY: PokemonBaseStats(hp=70, attack=45, defense=48, speed=35, special=60),
    Pokemon.CLOYSTER: PokemonBaseStats(hp=50, attack=95, defense=180, speed=70, special=85),
    Pokemon.CUBONE: PokemonBaseStats(hp=50, attack=50, defense=95, speed=35, special=40),
    Pokemon.DEWGONG: PokemonBaseStats(hp=90, attack=70, defense=80, speed=70, special=95),
    Pokemon.DIGLETT: PokemonBaseStats(hp=10, attack=55, defense=25, speed=95, special=45),
    Pokemon.DITTO: PokemonBaseStats(hp=48, attack=48, defense=48, speed=48, special=48),
    Pokemon.DODRIO: PokemonBaseStats(hp=60, attack=110, defense=70, speed=100, special=60),
    Pokemon.DODUO: PokemonBaseStats(hp=35, attack=85, defense=45, speed=75, special=35),
    Pokemon.DRAGONAIR: PokemonBaseStats(hp=61, attack=84, defense=65, speed=70, special=70),
    Pokemon.DRAGONITE: PokemonBaseStats(hp=91, attack=134, defense=95, speed=80, special=100),
    Pokemon.DRATINI: PokemonBaseStats(hp=41, attack=64, defense=45, speed=50, special=50),
    Pokemon.DROWZEE: PokemonBaseStats(hp=60, attack=48, defense=45, speed=42, special=90),
    Pokemon.DUGTRIO: PokemonBaseStats(hp=35, attack=80, defense=50, speed=120, special=70),
    Pokemon.EEVEE: PokemonBaseStats(hp=55, attack=55, defense=50, speed=55, special=65),
    Pokemon.EKANS: PokemonBaseStats(hp=35, attack=60, defense=44, speed=55, special=40),
    Pokemon.ELECTABUZZ: PokemonBaseStats(hp=65, attack=83, defense=57, speed=105, special=85),
    Pokemon.ELECTRODE: PokemonBaseStats(hp=60, attack=50, defense=70, speed=140, special=80),
    Pokemon.EXEGGCUTE: PokemonBaseStats(hp=60, attack=40, defense=80, speed=40, special=60),
    Pokemon.EXEGGUTOR: PokemonBaseStats(hp=95, attack=95, defense=85, speed=55, special=125),
    Pokemon.FARFETCHD: PokemonBaseStats(hp=52, attack=65, defense=55, speed=60, special=58),
    Pokemon.FEAROW: PokemonBaseStats(hp=65, attack=90, defense=65, speed=100, special=61),
    Pokemon.FLAREON: PokemonBaseStats(hp=65, attack=130, defense=60, speed=65, special=110),
    Pokemon.GASTLY: PokemonBaseStats(hp=30, attack=35, defense=30, speed=80, special=100),
    Pokemon.GENGAR: PokemonBaseStats(hp=60, attack=65, defense=60, speed=110, special=130),
    Pokemon.GEODUDE: PokemonBaseStats(hp=40, attack=80, defense=100, speed=20, special=30),
    Pokemon.GLOOM: PokemonBaseStats(hp=60, attack=65, defense=70, speed=40, special=85),
    Pokemon.GOLBAT: PokemonBaseStats(hp=75, attack=80, defense=70, speed=90, special=75),
    Pokemon.GOLDEEN: PokemonBaseStats(hp=45, attack=67, defense=60, speed=63, special=50),
    Pokemon.GOLDUCK: PokemonBaseStats(hp=80, attack=82, defense=78, speed=85, special=80),
    Pokemon.GOLEM: PokemonBaseStats(hp=80, attack=110, defense=130, speed=45, special=55),
    Pokemon.GRAVELER: PokemonBaseStats(hp=55, attack=95, defense=115, speed=35, special=45),
    Pokemon.GRIMER: PokemonBaseStats(hp=80, attack=80, defense=50, speed=25, special=40),
    Pokemon.GROWLITHE: PokemonBaseStats(hp=55, attack=70, defense=45, speed=60, special=50),
    Pokemon.GYARADOS: PokemonBaseStats(hp=95, attack=125, defense=79, speed=81, special=100),
    Pokemon.HAUNTER: PokemonBaseStats(hp=45, attack=50, defense=45, speed=95, special=115),
    Pokemon.HITMONCHAN: PokemonBaseStats(hp=50, attack=105, defense=79, speed=76, special=35),
    Pokemon.HITMONLEE: PokemonBaseStats(hp=50, attack=120, defense=53, speed=87, special=35),
    Pokemon.HORSEA: PokemonBaseStats(hp=30, attack=40, defense=70, speed=60, special=70),
    Pokemon.HYPNO: PokemonBaseStats(hp=85, attack=73, defense=70, speed=67, special=115),
    Pokemon.IVYSAUR: PokemonBaseStats(hp=60, attack=62, defense=63, speed=60, special=80),
    Pokemon.JIGGLYPUFF: PokemonBaseStats(hp=115, attack=45, defense=20, speed=20, special=25),
    Pokemon.JOLTEON: PokemonBaseStats(hp=65, attack=65, defense=60, speed=130, special=110),
    Pokemon.JYNX: PokemonBaseStats(hp=65, attack=50, defense=35, speed=95, special=95),
    Pokemon.KABUTO: PokemonBaseStats(hp=30, attack=80, defense=90, speed=55, special=45),
    Pokemon.KABUTOPS: PokemonBaseStats(hp=60, attack=115, defense=105, speed=80, special=70),
    Pokemon.KADABRA: PokemonBaseStats(hp=40, attack=35, defense=30, speed=105, special=120),
    Pokemon.KAKUNA: PokemonBaseStats(hp=45, attack=25, defense=50, speed=35, special=25),
    Pokemon.KANGASKHAN: PokemonBaseStats(hp=105, attack=95, defense=80, speed=90, special=40),
    Pokemon.KINGLER: PokemonBaseStats(hp=55, attack=130, defense=115, speed=75, special=50),
    Pokemon.KOFFING: PokemonBaseStats(hp=40, attack=65, defense=95, speed=35, special=60),
    Pokemon.KRABBY: PokemonBaseStats(hp=30, attack=105, defense=90, speed=50, special=25),
    Pokemon.LAPRAS: PokemonBaseStats(hp=130, attack=85, defense=80, speed=60, special=95),
    Pokemon.LICKITUNG: PokemonBaseStats(hp=90, attack=55, defense=75, speed=30, special=60),
    Pokemon.MACHAMP: PokemonBaseStats(hp=90, attack=130, defense=80, speed=55, special=65),
    Pokemon.MACHOKE: PokemonBaseStats(hp=80, attack=100, defense=70, speed=45, special=50),
    Pokemon.MACHOP: PokemonBaseStats(hp=70, attack=80, defense=50, speed=35, special=35),
    Pokemon.MAGIKARP: PokemonBaseStats(hp=20, attack=10, defense=55, speed=80, special=20),
    Pokemon.MAGMAR: PokemonBaseStats(hp=65, attack=95, defense=57, speed=93, special=85),
    Pokemon.MAGNEMITE: PokemonBaseStats(hp=25, attack=35, defense=70, speed=45, special=95),
    Pokemon.MAGNETON: PokemonBaseStats(hp=50, attack=60, defense=95, speed=70, special=120),
    Pokemon.MANKEY: PokemonBaseStats(hp=40, attack=80, defense=35, speed=70, special=35),
    Pokemon.MAROWAK: PokemonBaseStats(hp=60, attack=80, defense=110, speed=45, special=50),
    Pokemon.MEOWTH: PokemonBaseStats(hp=40, attack=45, defense=35, speed=90, special=40),
    Pokemon.METAPOD: PokemonBaseStats(hp=50, attack=20, defense=55, speed=30, special=25),
    Pokemon.MEW: PokemonBaseStats(hp=100, attack=100, defense=100, speed=100, special=100),
    Pokemon.MEWTWO: PokemonBaseStats(hp=106, attack=110, defense=90, speed=130, special=154),
    Pokemon.MOLTRES: PokemonBaseStats(hp=90, attack=100, defense=90, speed=90, special=125),
    Pokemon.MUK: PokemonBaseStats(hp=105, attack=105, defense=75, speed=50, special=65),
    Pokemon.NIDOKING: PokemonBaseStats(hp=81, attack=92, defense=77, speed=85, special=75),
    Pokemon.NIDOQUEEN: PokemonBaseStats(hp=90, attack=82, defense=87, speed=76, special=75),
    Pokemon.NIDORANF: PokemonBaseStats(hp=55, attack=47, defense=52, speed=41, special=40),
    Pokemon.NIDORANM: PokemonBaseStats(hp=46, attack=57, defense=40, speed=50, special=40),
    Pokemon.NIDORINA: PokemonBaseStats(hp=70, attack=62, defense=67, speed=56, special=55),
    Pokemon.NIDORINO: PokemonBaseStats(hp=61, attack=72, defense=57, speed=65, special=55),
    Pokemon.NINETALES: PokemonBaseStats(hp=73, attack=76, defense=75, speed=100, special=100),
    Pokemon.ODDISH: PokemonBaseStats(hp=45, attack=50, defense=55, speed=30, special=75),
    Pokemon.OMANYTE: PokemonBaseStats(hp=35, attack=40, defense=100, speed=35, special=90),
    Pokemon.OMASTAR: PokemonBaseStats(hp=70, attack=60, defense=125, speed=55, special=115),
    Pokemon.ONIX: PokemonBaseStats(hp=35, attack=45, defense=160, speed=70, special=30),
    Pokemon.PARAS: PokemonBaseStats(hp=35, attack=70, defense=55, speed=25, special=55),
    Pokemon.PARASECT: PokemonBaseStats(hp=60, attack=95, defense=80, speed=30, special=80),
    Pokemon.PERSIAN: PokemonBaseStats(hp=65, attack=70, defense=60, speed=115, special=65),
    Pokemon.PIDGEOT: PokemonBaseStats(hp=83, attack=80, defense=75, speed=91, special=70),
    Pokemon.PIDGEOTTO: PokemonBaseStats(hp=63, attack=60, defense=55, speed=71, special=50),
    Pokemon.PIDGEY: PokemonBaseStats(hp=40, attack=45, defense=40, speed=56, special=35),
    Pokemon.PIKACHU: PokemonBaseStats(hp=35, attack=55, defense=30, speed=90, special=50),
    Pokemon.PINSIR: PokemonBaseStats(hp=65, attack=125, defense=100, speed=85, special=55),
    Pokemon.POLIWAG: PokemonBaseStats(hp=40, attack=50, defense=40, speed=90, special=40),
    Pokemon.POLIWHIRL: PokemonBaseStats(hp=65, attack=65, defense=65, speed=90, special=50),
    Pokemon.POLIWRATH: PokemonBaseStats(hp=90, attack=85, defense=95, speed=70, special=70),
    Pokemon.PONYTA: PokemonBaseStats(hp=50, attack=85, defense=55, speed=90, special=65),
    Pokemon.PORYGON: PokemonBaseStats(hp=65, attack=60, defense=70, speed=40, special=75),
    Pokemon.PRIMEAPE: PokemonBaseStats(hp=65, attack=105, defense=60, speed=95, special=60),
    Pokemon.PSYDUCK: PokemonBaseStats(hp=50, attack=52, defense=48, speed=55, special=50),
    Pokemon.RAICHU: PokemonBaseStats(hp=60, attack=90, defense=55, speed=100, special=90),
    Pokemon.RAPIDASH: PokemonBaseStats(hp=65, attack=100, defense=70, speed=105, special=80),
    Pokemon.RATICATE: PokemonBaseStats(hp=55, attack=81, defense=60, speed=97, special=50),
    Pokemon.RATTATA: PokemonBaseStats(hp=30, attack=56, defense=35, speed=72, special=25),
    Pokemon.RHYDON: PokemonBaseStats(hp=105, attack=130, defense=120, speed=40, special=45),
    Pokemon.RHYHORN: PokemonBaseStats(hp=80, attack=85, defense=95, speed=25, special=30),
    Pokemon.SANDSHREW: PokemonBaseStats(hp=50, attack=75, defense=85, speed=40, special=30),
    Pokemon.SANDSLASH: PokemonBaseStats(hp=75, attack=100, defense=110, speed=65, special=55),
    Pokemon.SCYTHER: PokemonBaseStats(hp=70, attack=110, defense=80, speed=105, special=55),
    Pokemon.SEADRA: PokemonBaseStats(hp=55, attack=65, defense=95, speed=85, special=95),
    Pokemon.SEAKING: PokemonBaseStats(hp=80, attack=92, defense=65, speed=68, special=80),
    Pokemon.SEEL: PokemonBaseStats(hp=65, attack=45, defense=55, speed=45, special=70),
    Pokemon.SHELLDER: PokemonBaseStats(hp=30, attack=65, defense=100, speed=40, special=45),
    Pokemon.SLOWBRO: PokemonBaseStats(hp=95, attack=75, defense=110, speed=30, special=80),
    Pokemon.SLOWPOKE: PokemonBaseStats(hp=90, attack=65, defense=65, speed=15, special=40),
    Pokemon.SNORLAX: PokemonBaseStats(hp=160, attack=110, defense=65, speed=30, special=65),
    Pokemon.SPEAROW: PokemonBaseStats(hp=40, attack=60, defense=30, speed=70, special=31),
    Pokemon.SQUIRTLE: PokemonBaseStats(hp=44, attack=48, defense=65, speed=43, special=50),
    Pokemon.STARMIE: PokemonBaseStats(hp=60, attack=75, defense=85, speed=115, special=100),
    Pokemon.STARYU: PokemonBaseStats(hp=30, attack=45, defense=55, speed=85, special=70),
    Pokemon.TANGELA: PokemonBaseStats(hp=65, attack=55, defense=115, speed=60, special=100),
    Pokemon.TAUROS: PokemonBaseStats(hp=75, attack=100, defense=95, speed=110, special=70),
    Pokemon.TENTACOOL: PokemonBaseStats(hp=40, attack=40, defense=35, speed=70, special=100),
    Pokemon.TENTACRUEL: PokemonBaseStats(hp=80, attack=70, defense=65, speed=100, special=120),
    Pokemon.VAPOREON: PokemonBaseStats(hp=130, attack=65, defense=60, speed=65, special=110),
    Pokemon.VENOMOTH: PokemonBaseStats(hp=70, attack=65, defense=60, speed=90, special=90),
    Pokemon.VENONAT: PokemonBaseStats(hp=60, attack=55, defense=50, speed=45, special=40),
    Pokemon.VENUSAUR: PokemonBaseStats(hp=80, attack=82, defense=83, speed=80, special=100),
    Pokemon.VICTREEBEL: PokemonBaseStats(hp=80, attack=105, defense=65, speed=70, special=100),
    Pokemon.VILEPLUME: PokemonBaseStats(hp=75, attack=80, defense=85, speed=50, special=100),
    Pokemon.VOLTORB: PokemonBaseStats(hp=40, attack=30, defense=50, speed=100, special=55),
    Pokemon.VULPIX: PokemonBaseStats(hp=38, attack=41, defense=40, speed=65, special=65),
    Pokemon.WARTORTLE: PokemonBaseStats(hp=59, attack=63, defense=80, speed=58, special=65),
    Pokemon.WEEDLE: PokemonBaseStats(hp=40, attack=35, defense=30, speed=50, special=20),
    Pokemon.WEEPINBELL: PokemonBaseStats(hp=65, attack=90, defense=50, speed=55, special=85),
    Pokemon.WEEZING: PokemonBaseStats(hp=65, attack=90, defense=120, speed=60, special=85),
    Pokemon.WIGGLYTUFF: PokemonBaseStats(hp=140, attack=70, defense=45, speed=45, special=50),
    Pokemon.ZAPDOS: PokemonBaseStats(hp=90, attack=90, defense=85, speed=100, special=125),
    Pokemon.ZUBAT: PokemonBaseStats(hp=40, attack=45, defense=35, speed=55, special=40),
    Pokemon.MR_MIME: PokemonBaseStats(hp=40, attack=45, defense=65, speed=90, special=100),
}