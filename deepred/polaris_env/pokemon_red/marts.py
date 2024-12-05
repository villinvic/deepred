from enum import Enum
from typing import NamedTuple, List

from deepred.polaris_env.pokemon_red.enums import BagItem, Map

MART_ITEMS_ID_DICT = {'42@2,5': {'items': [4, 11, 15, 12], 'map': 'VIRIDIAN_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '56@2,5': {'items': [4, 20, 29, 11, 12, 14, 15], 'map': 'PEWTER_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '67@2,5': {'items': [4, 20, 30, 11, 12, 14, 15], 'map': 'CERULEAN_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '91@2,5': {'items': [4, 19, 13, 14, 15, 30], 'map': 'VERMILION_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '150@2,5': {'items': [3, 19, 53, 29, 56, 11, 12, 13, 15], 'map': 'LAVENDER_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '123@5,5': {'items': [3, 19, 53, 56, 11, 12, 13, 14, 15], 'map': 'CELADON_MART_2F', 'x': 5, 'y': 5, 'dir': 'up'}, '123@6,5': {'items': [], 'map': 'CELADON_MART_2F', 'x': 6, 'y': 5, 'dir': 'up'}, '125@5,5': {'items': [51, 32, 33, 34, 47], 'map': 'CELADON_MART_4F', 'x': 5, 'y': 5, 'dir': 'down'}, '136@5,5': {'items': [46, 55, 58, 65, 66, 67, 68], 'map': 'CELADON_MART_5F', 'x': 5, 'y': 5, 'dir': 'up'}, '136@6,5': {'items': [35, 36, 37, 38, 39], 'map': 'CELADON_MART_5F', 'x': 6, 'y': 5, 'dir': 'up'}, '152@2,5': {'items': [2, 3, 19, 53, 52, 56], 'map': 'FUCHSIA_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '172@2,5': {'items': [2, 3, 18, 57, 29, 52, 53], 'map': 'CINNABAR_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '180@2,5': {'items': [3, 18, 57, 29, 52, 53], 'map': 'SAFFRON_MART', 'x': 2, 'y': 5, 'dir': 'left'}, '174@2,5': {'items': [2, 3, 16, 17, 52, 53, 57], 'map': 'INDIGO_PLATEAU_LOBBY', 'x': 2, 'y': 5, 'dir': 'left'}}

class ClerkDirection(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MartInfo(NamedTuple):
    items: List[BagItem]
    map: Map
    x: int
    y: int
    clerk_direction: ClerkDirection


for name, martdict in MART_ITEMS_ID_DICT.items():
    items = "["
    for item in martdict["items"]:
        items += f"BagItem.{BagItem(item).name},"
    if len(martdict["items"]) > 0:
        items = items[:-2]
    items += "],"
    print(f"    '{name}': MartInfo(items={items}, map=Map.{Map[martdict['map']].name}, x={martdict['x']}, y={martdict['y']}, clerk_direction=ClerkDirection.{martdict['dir'].upper()})")