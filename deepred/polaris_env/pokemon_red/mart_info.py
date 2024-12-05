from enum import Enum
from typing import NamedTuple, List

from deepred.polaris_env.pokemon_red.enums import BagItem, Map, Orientation


class MartInfo(NamedTuple):
    items: List[BagItem] = []
    map: Map = Map.UNKNOWN
    x: int = 0
    y: int = 0
    clerk_orientation: Orientation = Orientation.UP


MartInfos = {
    'VIRIDIAN_MART@2,2,LEFT': MartInfo(
        items=[BagItem.POKE_BALL, BagItem.ANTIDOTE, BagItem.PARLYZ_HEAL, BagItem.BURN_HEAL], map=Map.VIRIDIAN_MART, x=2,
        y=5, clerk_orientation=Orientation.LEFT),
    'PEWTER_MART@2,2,LEFT': MartInfo(
        items=[BagItem.POKE_BALL, BagItem.POTION, BagItem.ESCAPE_ROPE, BagItem.ANTIDOTE, BagItem.BURN_HEAL,
               BagItem.AWAKENING, BagItem.PARLYZ_HEAL], map=Map.PEWTER_MART, x=2, y=5,
        clerk_orientation=Orientation.LEFT),
    'CERULEAN_MART@2,2,LEFT': MartInfo(
        items=[BagItem.POKE_BALL, BagItem.POTION, BagItem.REPEL, BagItem.ANTIDOTE, BagItem.BURN_HEAL, BagItem.AWAKENING,
               BagItem.PARLYZ_HEAL], map=Map.CERULEAN_MART, x=2, y=5, clerk_orientation=Orientation.LEFT),
    'VERMILION_MART@2,2,LEFT': MartInfo(
        items=[BagItem.POKE_BALL, BagItem.SUPER_POTION, BagItem.ICE_HEAL, BagItem.AWAKENING, BagItem.PARLYZ_HEAL,
               BagItem.REPEL], map=Map.VERMILION_MART, x=2, y=5, clerk_orientation=Orientation.LEFT),
    'LAVENDER_MART@2,2,LEFT': MartInfo(
        items=[BagItem.GREAT_BALL, BagItem.SUPER_POTION, BagItem.REVIVE, BagItem.ESCAPE_ROPE, BagItem.SUPER_REPEL,
               BagItem.ANTIDOTE, BagItem.BURN_HEAL, BagItem.ICE_HEAL, BagItem.PARLYZ_HEAL], map=Map.LAVENDER_MART, x=2,
        y=5, clerk_orientation=Orientation.LEFT),
    'CELADON_MART_2F@5,5,UP': MartInfo(
        items=[BagItem.GREAT_BALL, BagItem.SUPER_POTION, BagItem.REVIVE, BagItem.SUPER_REPEL, BagItem.ANTIDOTE,
               BagItem.BURN_HEAL, BagItem.ICE_HEAL, BagItem.AWAKENING, BagItem.PARLYZ_HEAL], map=Map.CELADON_MART_2F,
        x=5, y=5, clerk_orientation=Orientation.UP),
    'CELADON_MART_2F@6,6,UP': MartInfo(items=[], map=Map.CELADON_MART_2F, x=6, y=5, clerk_orientation=Orientation.UP),
    'CELADON_MART_4F@5,5,DOWN': MartInfo(
        items=[BagItem.POKE_DOLL, BagItem.FIRE_STONE, BagItem.THUNDER_STONE, BagItem.WATER_STONE, BagItem.LEAF_STONE],
        map=Map.CELADON_MART_4F, x=5, y=5, clerk_orientation=Orientation.DOWN),
    'CELADON_MART_5F@5,5,UP': MartInfo(
        items=[BagItem.X_ACCURACY, BagItem.GUARD_SPEC, BagItem.DIRE_HIT, BagItem.X_ATTACK, BagItem.X_DEFEND,
               BagItem.X_SPEED, BagItem.X_SPECIAL], map=Map.CELADON_MART_5F, x=5, y=5,
        clerk_orientation=Orientation.UP),
    'CELADON_MART_5F@6,6,UP': MartInfo(
        items=[BagItem.HP_UP, BagItem.PROTEIN, BagItem.IRON, BagItem.CARBOS, BagItem.CALCIUM], map=Map.CELADON_MART_5F,
        x=6, y=5, clerk_orientation=Orientation.UP),
    'FUCHSIA_MART@2,2,LEFT': MartInfo(
        items=[BagItem.ULTRA_BALL, BagItem.GREAT_BALL, BagItem.SUPER_POTION, BagItem.REVIVE, BagItem.FULL_HEAL,
               BagItem.SUPER_REPEL], map=Map.FUCHSIA_MART, x=2, y=5, clerk_orientation=Orientation.LEFT),
    'CINNABAR_MART@2,2,LEFT': MartInfo(
        items=[BagItem.ULTRA_BALL, BagItem.GREAT_BALL, BagItem.HYPER_POTION, BagItem.MAX_REPEL, BagItem.ESCAPE_ROPE,
               BagItem.FULL_HEAL, BagItem.REVIVE], map=Map.CINNABAR_MART, x=2, y=5, clerk_orientation=Orientation.LEFT),
    'SAFFRON_MART@2,2,LEFT': MartInfo(
        items=[BagItem.GREAT_BALL, BagItem.HYPER_POTION, BagItem.MAX_REPEL, BagItem.ESCAPE_ROPE, BagItem.FULL_HEAL,
               BagItem.REVIVE], map=Map.SAFFRON_MART, x=2, y=5, clerk_orientation=Orientation.LEFT),
    'INDIGO_PLATEAU_LOBBY@2,2,LEFT': MartInfo(
        items=[BagItem.ULTRA_BALL, BagItem.GREAT_BALL, BagItem.FULL_RESTORE, BagItem.MAX_POTION, BagItem.FULL_HEAL,
               BagItem.REVIVE, BagItem.MAX_REPEL], map=Map.INDIGO_PLATEAU_LOBBY, x=2, y=5,
        clerk_orientation=Orientation.LEFT),

}
