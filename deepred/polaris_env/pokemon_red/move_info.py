from typing import NamedTuple, List

import numpy as np

from deepred.polaris_env.pokemon_red.enums import Move, FixedPokemonType

class MoveEffect:
    value: int = 1

    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        actual_power = move.actual_power(pokemon_types)

        return actual_power * move.accuracy


class NoEffect(MoveEffect):
    pass


class TwoToFiveHitsEffect(MoveEffect):

    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        actual_power = move.actual_power(pokemon_types)

        expected_num_hits = 2 + (
            move.accuracy * (1-move.accuracy)
            + 2 * (move.accuracy ** 2) * (1 - move.accuracy)
            + 3 * move.accuracy ** 3
        )
        return expected_num_hits * actual_power


class TwoHitsEffect(MoveEffect):

    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        return super().power_function(move, pokemon_types) * 2


class PayDayEffect(MoveEffect):
    value = 2


class BurnEffect10(MoveEffect):
    value = 3
        

class FreezeEffect10(MoveEffect):
    value = 3


class ParalisedEffect10(MoveEffect):
    value = 3


class ParalisedEffect30(MoveEffect):
    value = 9

        
class BURN_SIDE_EFFECT30(MoveEffect):
    value = 9


class ConfusionEffect(MoveEffect):
    value = 30


class OHKOEffect(MoveEffect):
    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        return 150


class ChargeEffect(MoveEffect):
    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        return super().power_function(move, pokemon_types) * 0.65


class AttackUp2Effect(MoveEffect):
    # TODO: add stat modifiers to observations
    value = 30
    
    
class SwitchEffect(MoveEffect):
    pass
    

class FlyEffect(MoveEffect):
    pass
    

class TrappingEffect(MoveEffect):
    value = 5

    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        return super().power_function(move, pokemon_types) * 1.5
    

class FlinchEffect30(MoveEffect):
    value = 9
    
    
class JumpKickEffect(MoveEffect):
    def power_function(self, move: "MoveInfo", pokemon_types: List[FixedPokemonType]):
        return super().power_function(move, pokemon_types) * 0.8

# etc for other effects...
# TODO


class MoveInfo(NamedTuple):
    effect: MoveEffect
    type_id : FixedPokemonType
    raw_power: int
    power: float
    accuracy: int
    pp: int

    def actual_power(
            self,
            pokemon_types: List[FixedPokemonType]
    ):
        """
        Computes the actual power based on the pokemon types.
        """
        mult = 1.
        if self.type_id in pokemon_types:
            mult += 0.5
        return self.raw_power * mult

    def expected_power(
            self,
            pokemon_types: List[FixedPokemonType]
    ):
        """
        Computes the expected power based on the pokemon types.
        """
        if isinstance(self.effect, str):
            # TODO
            return NoEffect().power_function(self, pokemon_types)
        return self.effect().power_function(self, pokemon_types)


    def utility(
            self,
            pokemon_types: List[int],
            known_moves: List["MoveInfo"],
    ) -> list[float]:
        """
        Checks the move utility (should we replace a known move with this one ?)
        Stronger moves, in power, should be learned
        If we have moves with similar effects, we should replace the move with the stronger power.

        TODO


        :return: list of values, each describing the utility of replacing the corresponding known move with the new one.
        """
        utilities = []
        existing_types = {move.type_id for move in known_moves}
        existing_effects = {type(move.effect) for move in known_moves}

        # we should ensure that the pokemon has at least two hitting moves

        move_powers = [move.expected_power(pokemon_types) for move in known_moves]
        num_hiting_moves = len([move.raw_power > 0 for move in known_moves])

        for move in known_moves:
            utility = 0.0
            new_move_power = self.expected_power(pokemon_types)
            move_power = move.expected_power(pokemon_types)

            if self.raw_power == 0 and num_hiting_moves >= 2:
                if isinstance(self.effect, type(move.effect)):
                    utility = self.effect.value / (move_power + move.effect.utility)

            # Check if the new move has the same type as the known move


        return utilities



MovesInfo = {
    Move.NO_MOVE: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=0, power=0, accuracy=0, pp=0),
    Move.POUND: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=40, power=40.0, accuracy=100, pp=35),
    Move.KARATE_CHOP: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=50, power=50.0, accuracy=100, pp=25),
    Move.DOUBLESLAP: MoveInfo(effect=TwoToFiveHitsEffect, type_id=FixedPokemonType.NORMAL, raw_power=15, power=37.8984375,
                              accuracy=85, pp=10),
    Move.COMET_PUNCH: MoveInfo(effect=TwoToFiveHitsEffect, type_id=FixedPokemonType.NORMAL, raw_power=18, power=48.7265625,
                               accuracy=85, pp=15),
    Move.MEGA_PUNCH: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=80, power=77.0, accuracy=85, pp=20),
    Move.PAY_DAY: MoveInfo(effect=PayDayEffect, type_id=FixedPokemonType.NORMAL, raw_power=40, power=60.0, accuracy=100, pp=20),
    Move.FIRE_PUNCH: MoveInfo(effect=BurnEffect10, type_id=FixedPokemonType.FIRE, raw_power=75, power=79.6875, accuracy=100, pp=15),
    Move.ICE_PUNCH: MoveInfo(effect=FreezeEffect10, type_id=FixedPokemonType.ICE, raw_power=75, power=89.0625, accuracy=100, pp=15),
    Move.THUNDERPUNCH: MoveInfo(effect=ParalisedEffect10, type_id=FixedPokemonType.ELECTRIC, raw_power=75, power=89.0625, accuracy=100,
                                pp=15),
    Move.SCRATCH: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=40, power=40.0, accuracy=100, pp=35),
    Move.VICEGRIP: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=55, power=55.0, accuracy=100, pp=30),
    Move.GUILLOTINE: MoveInfo(effect=OHKOEffect, type_id=FixedPokemonType.NORMAL, raw_power=1, power=40.8890625, accuracy=30, pp=5),
    Move.RAZOR_WIND: MoveInfo(effect=ChargeEffect, type_id=FixedPokemonType.NORMAL, raw_power=80, power=52.5, accuracy=75, pp=10),
    Move.SWORDS_DANCE: MoveInfo(effect=AttackUp2Effect, type_id=FixedPokemonType.NORMAL, raw_power=0, power=20.0, accuracy=100, pp=30),
    Move.CUT: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=50, power=49.375, accuracy=95, pp=30),
    Move.GUST: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=40, power=40.0, accuracy=100, pp=35),
    Move.WING_ATTACK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.FLYING, raw_power=35, power=35.0, accuracy=100, pp=35),
    Move.WHIRLWIND: MoveInfo(effect=SwitchEffect, type_id=FixedPokemonType.NORMAL, raw_power=0, power=0.0, accuracy=85,
                             pp=20),
    Move.FLY: MoveInfo(effect=FlyEffect, type_id=FixedPokemonType.FLYING, raw_power=70, power=64.8046875, accuracy=95, pp=15),
    Move.BIND: MoveInfo(effect=TrappingEffect, type_id=FixedPokemonType.NORMAL, raw_power=15, power=56.25, accuracy=75, pp=20),
    Move.SLAM: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=80, power=75.0, accuracy=75, pp=20),
    Move.VINE_WHIP: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.GRASS, raw_power=35, power=30.625, accuracy=100,
                             pp=10),
    Move.STOMP: MoveInfo(effect=FlinchEffect30, type_id=FixedPokemonType.NORMAL, raw_power=65, power=105.0, accuracy=100, pp=20),
    Move.DOUBLE_KICK: MoveInfo(effect=TwoHitsEffect, type_id=FixedPokemonType.FIGHTING, raw_power=30, power=60.0, accuracy=100, pp=30),
    Move.MEGA_KICK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=120, power=91.40625, accuracy=75,
                             pp=5),
    Move.JUMP_KICK: MoveInfo(effect=JumpKickEffect, type_id=FixedPokemonType.FIGHTING, raw_power=70, power=62.212500000000006, accuracy=95,
                             pp=25),
    Move.ROLLING_KICK: MoveInfo(effect=FlinchEffect30, type_id=FixedPokemonType.FIGHTING, raw_power=60, power=90.234375, accuracy=85,
                                pp=15),
    Move.SAND_ATTACK: MoveInfo(effect="ACCURACY_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=9.375, accuracy=100,
                               pp=15),
    Move.HEADBUTT: MoveInfo(effect=FlinchEffect30, type_id=FixedPokemonType.NORMAL, raw_power=70, power=103.125, accuracy=100, pp=15),
    Move.HORN_ATTACK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=65, power=65.0, accuracy=100, pp=25),
    Move.FURY_ATTACK: MoveInfo(effect="TWO_TO_FIVE_ATTACKS_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=15, power=43.3125, accuracy=85,
                               pp=20),
    Move.HORN_DRILL: MoveInfo(effect=OHKOEffect, type_id=FixedPokemonType.NORMAL, raw_power=1, power=40.8890625, accuracy=30, pp=5),
    Move.TACKLE: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=35, power=34.5625, accuracy=95, pp=35),
    Move.BODY_SLAM: MoveInfo(effect=ParalisedEffect30, type_id=FixedPokemonType.NORMAL, raw_power=85, power=98.4375, accuracy=100,
                             pp=15),
    Move.WRAP: MoveInfo(effect=TrappingEffect, type_id=FixedPokemonType.NORMAL, raw_power=15, power=57.75, accuracy=85, pp=20),
    Move.TAKE_DOWN: MoveInfo(effect="RECOIL_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=90, power=77.9625, accuracy=85, pp=20),
    Move.THRASH: MoveInfo(effect="THRASH_PETAL_DANCE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=90, power=81.0, accuracy=100, pp=20),
    Move.DOUBLE_EDGE: MoveInfo(effect="RECOIL_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=100, power=84.375, accuracy=100, pp=15),
    Move.TAIL_WHIP: MoveInfo(effect="DEFENSE_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.POISON_STING: MoveInfo(effect="POISON_SIDE_EFFECT1", type_id=FixedPokemonType.POISON, raw_power=15, power=25.0, accuracy=100, pp=35),
    Move.TWINEEDLE: MoveInfo(effect="TWINEEDLE_EFFECT", type_id=FixedPokemonType.BUG, raw_power=25, power=62.0, accuracy=100, pp=20),
    Move.PIN_MISSILE: MoveInfo(effect="TWO_TO_FIVE_ATTACKS_EFFECT", type_id=FixedPokemonType.BUG, raw_power=14, power=40.425000000000004,
                               accuracy=85, pp=20),
    Move.LEER: MoveInfo(effect="DEFENSE_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.BITE: MoveInfo(effect="FLINCH_SIDE_EFFECT1", type_id=FixedPokemonType.NORMAL, raw_power=60, power=80.0, accuracy=100, pp=25),
    Move.GROWL: MoveInfo(effect="ATTACK_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=40),
    Move.ROAR: MoveInfo(effect=SwitchEffect, type_id=FixedPokemonType.NORMAL, raw_power=0, power=0.0, accuracy=100, pp=20),
    Move.SING: MoveInfo(effect="SLEEP_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=16.640625, accuracy=55, pp=15),
    Move.SUPERSONIC: MoveInfo(effect="CONFUSION_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=8.875, accuracy=55, pp=20),
    Move.SONICBOOM: MoveInfo(effect="SPECIAL_DAMAGE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=1, power=39.975, accuracy=90, pp=20),
    Move.DISABLE: MoveInfo(effect="DISABLE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=17.75, accuracy=55, pp=20),
    Move.ACID: MoveInfo(effect="DEFENSE_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.POISON, raw_power=40, power=50.0, accuracy=100, pp=30),
    Move.EMBER: MoveInfo(effect=BurnEffect10, type_id=FixedPokemonType.FIRE, raw_power=40, power=50.0, accuracy=100, pp=25),
    Move.FLAMETHROWER: MoveInfo(effect=BurnEffect10, type_id=FixedPokemonType.FIRE, raw_power=95, power=98.4375, accuracy=100,
                                pp=15),
    Move.MIST: MoveInfo(effect="MIST_EFFECT", type_id=FixedPokemonType.ICE, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.WATER_GUN: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.WATER, raw_power=40, power=40.0, accuracy=100, pp=25),
    Move.HYDRO_PUMP: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.WATER, raw_power=120, power=92.625, accuracy=80,
                              pp=5),
    Move.SURF: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.WATER, raw_power=95, power=89.0625, accuracy=100, pp=15),
    Move.ICE_BEAM: MoveInfo(effect=FreezeEffect10, type_id=FixedPokemonType.ICE, raw_power=95, power=100.625, accuracy=100, pp=10),
    Move.BLIZZARD: MoveInfo(effect=FreezeEffect10, type_id=FixedPokemonType.ICE, raw_power=120, power=110.90624999999999,
                            accuracy=90, pp=5),
    Move.PSYBEAM: MoveInfo(effect="CONFUSION_SIDE_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=65, power=95.0, accuracy=100, pp=20),
    Move.BUBBLEBEAM: MoveInfo(effect="SPEED_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.WATER, raw_power=65, power=75.0, accuracy=100,
                              pp=20),
    Move.AURORA_BEAM: MoveInfo(effect="ATTACK_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.ICE, raw_power=65, power=75.0, accuracy=100,
                               pp=20),
    Move.HYPER_BEAM: MoveInfo(effect="HYPER_BEAM_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=150, power=88.725, accuracy=90, pp=5),
    Move.PECK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.FLYING, raw_power=35, power=35.0, accuracy=100, pp=35),
    Move.DRILL_PECK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.FLYING, raw_power=80, power=80.0, accuracy=100, pp=20),
    Move.SUBMISSION: MoveInfo(effect="RECOIL_EFFECT", type_id=FixedPokemonType.FIGHTING, raw_power=80, power=68.39999999999999, accuracy=80,
                              pp=25),
    Move.LOW_KICK: MoveInfo(effect=FlinchEffect30, type_id=FixedPokemonType.FIGHTING, raw_power=50, power=87.75, accuracy=90, pp=20),
    Move.COUNTER: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.FIGHTING, raw_power=1, power=1.0, accuracy=100, pp=20),
    Move.SEISMIC_TOSS: MoveInfo(effect="SPECIAL_DAMAGE_EFFECT", type_id=FixedPokemonType.FIGHTING, raw_power=1, power=41.0, accuracy=100,
                                pp=20),
    Move.STRENGTH: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=80, power=75.0, accuracy=100, pp=15),
    Move.ABSORB: MoveInfo(effect="DRAIN_HP_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=20, power=30.0, accuracy=100, pp=20),
    Move.MEGA_DRAIN: MoveInfo(effect="DRAIN_HP_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=40, power=52.5, accuracy=100, pp=10),
    Move.LEECH_SEED: MoveInfo(effect="LEECH_SEED_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=0, power=34.125, accuracy=90, pp=10),
    Move.GROWTH: MoveInfo(effect="SPECIAL_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=40),
    Move.RAZOR_LEAF: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.GRASS, raw_power=55, power=54.3125, accuracy=95,
                              pp=25),
    Move.SOLARBEAM: MoveInfo(effect=ChargeEffect, type_id=FixedPokemonType.GRASS, raw_power=120, power=84.0, accuracy=100, pp=10),
    Move.POISONPOWDER: MoveInfo(effect="POISON_EFFECT", type_id=FixedPokemonType.POISON, raw_power=0, power=18.75, accuracy=75, pp=35),
    Move.STUN_SPORE: MoveInfo(effect="PARALYZE_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=0, power=28.125, accuracy=75, pp=30),
    Move.SLEEP_POWDER: MoveInfo(effect="SLEEP_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=0, power=17.578125, accuracy=75, pp=15),
    Move.PETAL_DANCE: MoveInfo(effect="THRASH_PETAL_DANCE_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=70, power=63.0, accuracy=100,
                               pp=20),
    Move.STRING_SHOT: MoveInfo(effect="SPEED_DOWN1_EFFECT", type_id=FixedPokemonType.BUG, raw_power=0, power=9.875, accuracy=95, pp=40),
    Move.DRAGON_RAGE: MoveInfo(effect="SPECIAL_DAMAGE_EFFECT", type_id=FixedPokemonType.DRAGON, raw_power=1, power=35.875, accuracy=100,
                               pp=10),
    Move.FIRE_SPIN: MoveInfo(effect=TrappingEffect, type_id=FixedPokemonType.FIRE, raw_power=15, power=52.03125, accuracy=70, pp=15),
    Move.THUNDERSHOCK: MoveInfo(effect=ParalisedEffect10, type_id=FixedPokemonType.ELECTRIC, raw_power=40, power=60.0, accuracy=100,
                                pp=30),
    Move.THUNDERBOLT: MoveInfo(effect=ParalisedEffect10, type_id=FixedPokemonType.ELECTRIC, raw_power=95, power=107.8125, accuracy=100,
                               pp=15),
    Move.THUNDER_WAVE: MoveInfo(effect="PARALYZE_EFFECT", type_id=FixedPokemonType.ELECTRIC, raw_power=0, power=30.0, accuracy=100, pp=20),
    Move.THUNDER: MoveInfo(effect=ParalisedEffect10, type_id=FixedPokemonType.ELECTRIC, raw_power=120, power=113.31250000000001,
                           accuracy=70, pp=10),
    Move.ROCK_THROW: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.ROCK, raw_power=50, power=42.7734375, accuracy=65,
                              pp=15),
    Move.EARTHQUAKE: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.GROUND, raw_power=100, power=87.5, accuracy=100, pp=10),
    Move.FISSURE: MoveInfo(effect=OHKOEffect, type_id=FixedPokemonType.GROUND, raw_power=1, power=40.8890625, accuracy=30, pp=5),
    Move.DIG: MoveInfo(effect=ChargeEffect, type_id=FixedPokemonType.GROUND, raw_power=100, power=70.0, accuracy=100, pp=10),
    Move.TOXIC: MoveInfo(effect="POISON_EFFECT", type_id=FixedPokemonType.POISON, raw_power=0, power=16.84375, accuracy=85, pp=10),
    Move.CONFUSION: MoveInfo(effect="CONFUSION_SIDE_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=50, power=80.0, accuracy=100, pp=25),
    Move.PSYCHIC_M: MoveInfo(effect="SPECIAL_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=90, power=87.5, accuracy=100,
                             pp=10),
    Move.HYPNOSIS: MoveInfo(effect="SLEEP_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=18.0, accuracy=60, pp=20),
    Move.MEDITATE: MoveInfo(effect="ATTACK_UP1_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=10.0, accuracy=100, pp=40),
    Move.AGILITY: MoveInfo(effect="SPEED_UP2_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=30.0, accuracy=100, pp=30),
    Move.QUICK_ATTACK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=40, power=40.0, accuracy=100,
                                pp=30),
    Move.RAGE: MoveInfo(effect="RAGE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=20, power=40.0, accuracy=100, pp=20),
    Move.TELEPORT: MoveInfo(effect=SwitchEffect, type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=0.0, accuracy=100,
                            pp=20),
    Move.NIGHT_SHADE: MoveInfo(effect="SPECIAL_DAMAGE_EFFECT", type_id=FixedPokemonType.GHOST, raw_power=0, power=37.5, accuracy=100, pp=15),
    Move.MIMIC: MoveInfo(effect="MIMIC_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=8.75, accuracy=100, pp=10),
    Move.SCREECH: MoveInfo(effect="DEFENSE_DOWN2_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=19.25, accuracy=85, pp=40),
    Move.DOUBLE_TEAM: MoveInfo(effect="EVASION_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=18.75, accuracy=100, pp=15),
    Move.RECOVER: MoveInfo(effect="HEAL_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=20.0, accuracy=100, pp=20),
    Move.HARDEN: MoveInfo(effect="DEFENSE_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.MINIMIZE: MoveInfo(effect="EVASION_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=20.0, accuracy=100, pp=20),
    Move.SMOKESCREEN: MoveInfo(effect="ACCURACY_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=20),
    Move.CONFUSE_RAY: MoveInfo(effect="CONFUSION_EFFECT", type_id=FixedPokemonType.GHOST, raw_power=0, power=8.75, accuracy=100, pp=10),
    Move.WITHDRAW: MoveInfo(effect="DEFENSE_UP1_EFFECT", type_id=FixedPokemonType.WATER, raw_power=0, power=10.0, accuracy=100, pp=40),
    Move.DEFENSE_CURL: MoveInfo(effect="DEFENSE_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=40),
    Move.BARRIER: MoveInfo(effect="DEFENSE_UP2_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=20.0, accuracy=100, pp=30),
    Move.LIGHT_SCREEN: MoveInfo(effect="LIGHT_SCREEN_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=20.0, accuracy=100, pp=30),
    Move.HAZE: MoveInfo(effect="HAZE_EFFECT", type_id=FixedPokemonType.ICE, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.REFLECT: MoveInfo(effect="REFLECT_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=20.0, accuracy=100, pp=20),
    Move.FOCUS_ENERGY: MoveInfo(effect="FOCUS_ENERGY_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=20.0, accuracy=100, pp=30),
    Move.BIDE: MoveInfo(effect="BIDE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=35.0, accuracy=100, pp=10),
    Move.METRONOME: MoveInfo(effect="METRONOME_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=52.5, accuracy=100, pp=10),
    Move.MIRROR_MOVE: MoveInfo(effect="MIRROR_MOVE_EFFECT", type_id=FixedPokemonType.FLYING, raw_power=0, power=30.0, accuracy=100, pp=20),
    Move.SELFDESTRUCT: MoveInfo(effect="EXPLODE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=130, power=34.125, accuracy=100, pp=5),
    Move.EGG_BOMB: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=100, power=82.03125, accuracy=75,
                            pp=10),
    Move.LICK: MoveInfo(effect=ParalisedEffect30, type_id=FixedPokemonType.GHOST, raw_power=20, power=40.0, accuracy=100, pp=30),
    Move.SMOG: MoveInfo(effect="POISON_SIDE_EFFECT2", type_id=FixedPokemonType.POISON, raw_power=20, power=46.25, accuracy=70, pp=20),
    Move.SLUDGE: MoveInfo(effect="POISON_SIDE_EFFECT2", type_id=FixedPokemonType.POISON, raw_power=65, power=95.0, accuracy=100, pp=20),
    Move.BONE_CLUB: MoveInfo(effect="FLINCH_SIDE_EFFECT1", type_id=FixedPokemonType.GROUND, raw_power=65, power=81.8125, accuracy=85, pp=20),
    Move.FIRE_BLAST: MoveInfo(effect="BURN_SIDE_EFFECT2", type_id=FixedPokemonType.FIRE, raw_power=120, power=101.66406250000001,
                              accuracy=85, pp=5),
    Move.WATERFALL: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.WATER, raw_power=80, power=75.0, accuracy=100, pp=15),
    Move.CLAMP: MoveInfo(effect=TrappingEffect, type_id=FixedPokemonType.WATER, raw_power=35, power=114.84375, accuracy=75, pp=10),
    Move.SWIFT: MoveInfo(effect="SWIFT_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=60, power=80.0, accuracy=100, pp=20),
    Move.SKULL_BASH: MoveInfo(effect=ChargeEffect, type_id=FixedPokemonType.NORMAL, raw_power=100, power=75.0, accuracy=100, pp=15),
    Move.SPIKE_CANNON: MoveInfo(effect="TWO_TO_FIVE_ATTACKS_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=20, power=56.25, accuracy=100,
                                pp=15),
    Move.CONSTRICT: MoveInfo(effect="SPEED_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=10, power=20.0, accuracy=100, pp=35),
    Move.AMNESIA: MoveInfo(effect="SPECIAL_UP2_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=20.0, accuracy=100, pp=20),
    Move.KINESIS: MoveInfo(effect="ACCURACY_DOWN1_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=8.90625, accuracy=80, pp=15),
    Move.SOFTBOILED: MoveInfo(effect="HEAL_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=17.5, accuracy=100, pp=10),
    Move.HI_JUMP_KICK: MoveInfo(effect=JumpKickEffect, type_id=FixedPokemonType.FIGHTING, raw_power=85, power=74.1, accuracy=90, pp=20),
    Move.GLARE: MoveInfo(effect="PARALYZE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=28.125, accuracy=75, pp=30),
    Move.DREAM_EATER: MoveInfo(effect="DREAM_EATER_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=100, power=30.9375, accuracy=100,
                               pp=15),
    Move.POISON_GAS: MoveInfo(effect="POISON_EFFECT", type_id=FixedPokemonType.POISON, raw_power=0, power=17.75, accuracy=55, pp=40),
    Move.BARRAGE: MoveInfo(effect="TWO_TO_FIVE_ATTACKS_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=15, power=43.3125, accuracy=85,
                           pp=20),
    Move.LEECH_LIFE: MoveInfo(effect="DRAIN_HP_EFFECT", type_id=FixedPokemonType.BUG, raw_power=20, power=28.125, accuracy=100, pp=15),
    Move.LOVELY_KISS: MoveInfo(effect="SLEEP_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=16.40625, accuracy=75, pp=10),
    Move.SKY_ATTACK: MoveInfo(effect=ChargeEffect, type_id=FixedPokemonType.FLYING, raw_power=140, power=88.725, accuracy=90, pp=5),
    Move.TRANSFORM: MoveInfo(effect="TRANSFORM_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=8.75, accuracy=100, pp=10),
    Move.BUBBLE: MoveInfo(effect="SPEED_DOWN_SIDE_EFFECT", type_id=FixedPokemonType.WATER, raw_power=20, power=30.0, accuracy=100, pp=30),
    Move.DIZZY_PUNCH: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=70, power=61.25, accuracy=100,
                               pp=10),
    Move.SPORE: MoveInfo(effect="SLEEP_EFFECT", type_id=FixedPokemonType.GRASS, raw_power=0, power=18.75, accuracy=100, pp=15),
    Move.FLASH: MoveInfo(effect="ACCURACY_DOWN1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=9.25, accuracy=70, pp=20),
    Move.PSYWAVE: MoveInfo(effect="SPECIAL_DAMAGE_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=1, power=36.515625, accuracy=80,
                           pp=15),
    Move.SPLASH: MoveInfo(effect="SPLASH_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=0.0, accuracy=100, pp=40),
    Move.ACID_ARMOR: MoveInfo(effect="DEFENSE_UP2_EFFECT", type_id=FixedPokemonType.POISON, raw_power=0, power=20.0, accuracy=100, pp=40),
    Move.CRABHAMMER: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.WATER, raw_power=90, power=75.796875, accuracy=85,
                              pp=10),
    Move.EXPLOSION: MoveInfo(effect="EXPLODE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=170, power=45.5, accuracy=100, pp=5),
    Move.FURY_SWIPES: MoveInfo(effect="TWO_TO_FIVE_ATTACKS_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=18, power=48.09375,
                               accuracy=80, pp=15),
    Move.BONEMERANG: MoveInfo(effect=TwoHitsEffect, type_id=FixedPokemonType.GROUND, raw_power=50, power=85.3125, accuracy=90, pp=10),
    Move.REST: MoveInfo(effect="HEAL_EFFECT", type_id=FixedPokemonType.PSYCHIC_TYPE, raw_power=0, power=17.5, accuracy=100, pp=10),
    Move.ROCK_SLIDE: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.ROCK, raw_power=75, power=63.984375, accuracy=90,
                              pp=10),
    Move.HYPER_FANG: MoveInfo(effect="FLINCH_SIDE_EFFECT1", type_id=FixedPokemonType.NORMAL, raw_power=80, power=91.40625, accuracy=90,
                              pp=15),
    Move.SHARPEN: MoveInfo(effect="ATTACK_UP1_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.CONVERSION: MoveInfo(effect="CONVERSION_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=10.0, accuracy=100, pp=30),
    Move.TRI_ATTACK: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=80, power=70.0, accuracy=100, pp=10),
    Move.SUPER_FANG: MoveInfo(effect="SUPER_FANG_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=1, power=34.978125, accuracy=90, pp=10),
    Move.SLASH: MoveInfo(effect=NoEffect, type_id=FixedPokemonType.NORMAL, raw_power=70, power=70.0, accuracy=100, pp=20),
    Move.SUBSTITUTE: MoveInfo(effect="SUBSTITUTE_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=0, power=35.0, accuracy=100, pp=10),
    Move.STRUGGLE: MoveInfo(effect="RECOIL_EFFECT", type_id=FixedPokemonType.NORMAL, raw_power=50, power=39.375, accuracy=100, pp=10),

}