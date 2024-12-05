from enum import IntEnum

from deepred.polaris_env.enums import BattleKind, RamLocation
from deepred.polaris_env.gamestate import GameState

CHARMAP = {'<NULL>': 0, '<PAGE>': 73, '<PKMN>': 74, '<_CONT>': 75, '<SCROLL>': 76, '<NEXT>': 78, '<LINE>': 79, '@': 80, '<PARA>': 81, '<PLAYER>': 82, '<RIVAL>': 83, '#': 84, '<CONT>': 85, '<……>': 86, '<DONE>': 87, '<PROMPT>': 88, '<TARGET>': 89, '<USER>': 90, '<PC>': 91, '<TM>': 92, '<TRAINER>': 93, '<ROCKET>': 94, '<DEXEND>': 95, '<BOLD_A>': 96, '<BOLD_B>': 97, '<BOLD_C>': 98, '<BOLD_D>': 99, '<BOLD_E>': 100, '<BOLD_F>': 101, '<BOLD_G>': 102, '<BOLD_H>': 103, '<BOLD_I>': 104, '<BOLD_V>': 105, '<BOLD_S>': 106, '<BOLD_L>': 107, '<BOLD_M>': 108, '<COLON>': 109, 'ぃ': 110, 'ぅ': 111, '‘': 112, '’': 113, '“': 114, '”': 115, '·': 116, '…': 117, 'ぁ': 118, 'ぇ': 119, 'ぉ': 120, '┌': 121, '─': 122, '┐': 123, '│': 124, '└': 125, '┘': 126, '<LV>': 110, '<to>': 112, '『': 114, '<ID>': 115, '№': 116, '′': 96, '″': 97, '<BOLD_P>': 114, '▲': 237, '<ED>': 240, 'A': 128, 'B': 129, 'C': 130, 'D': 131, 'E': 132, 'F': 133, 'G': 134, 'H': 135, 'I': 136, 'J': 137, 'K': 138, 'L': 139, 'M': 140, 'N': 141, 'O': 142, 'P': 143, 'Q': 144, 'R': 145, 'S': 146, 'T': 147, 'U': 148, 'V': 149, 'W': 150, 'X': 151, 'Y': 152, 'Z': 153, '(': 154, ')': 155, ':': 156, ';': 157, '[': 158, ']': 159, 'a': 160, 'b': 161, 'c': 162, 'd': 163, 'e': 164, 'f': 165, 'g': 166, 'h': 167, 'i': 168, 'j': 169, 'k': 170, 'l': 171, 'm': 172, 'n': 173, 'o': 174, 'p': 175, 'q': 176, 'r': 177, 's': 178, 't': 179, 'u': 180, 'v': 181, 'w': 182, 'x': 183, 'y': 184, 'z': 185, 'é': 186, "'d": 187, "'l": 188, "'s": 189, "'t": 190, "'v": 191, "'": 224, '<PK>': 225, '<MN>': 226, '-': 227, "'r": 228, "'m": 229, '?': 230, '!': 231, '.': 232, 'ァ': 233, 'ゥ': 234, 'ェ': 235, '▷': 236, '▶': 237, '▼': 238, '♂': 239, '¥': 240, '×': 241, '<DOT>': 242, '/': 243, ',': 244, '♀': 245, '0': 246, '1': 247, '2': 248, '3': 249, '4': 250, '5': 251, '6': 252, '7': 253, '8': 254, '9': 255}

class BattleState(IntEnum):
    ACTIONABLE = 0
    NICKNAME = 1
    SWITCH = 2
    LEARN_MOVE = 3
    REPLACE_MOVE = 4
    ABANDON_MOVE = 5
    NOT_IN_BATTLE = 6
    OTHER = 7


def parse_battle_state(gamestate: GameState) -> BattleState:
    """
    TODO: improve readability ?
    Returns the state of the game in a battle, helps us skip frames or automate swapping/etc.
    """
    read = gamestate._read
    textbox_id = gamestate.textbox_id

    if not gamestate.is_in_battle:
        return BattleState.NOT_IN_BATTLE

    # Weird menu state
    if gamestate.open_menu and read(RamLocation.TILE_MAP_BASE + 12 * 20) != CHARMAP["┌"]:
        return BattleState.ACTIONABLE

    if gamestate.battle_kind == BattleKind.SAFARI:
        if textbox_id == 0x1b and \
                read(RamLocation.TILE_MAP_BASE + 14 * 20 + 14) == CHARMAP["B"] and \
                read(RamLocation.TILE_MAP_BASE + 14 * 20 + 15) == CHARMAP["A"]:
            return BattleState.ACTIONABLE
        elif textbox_id == 0x14 and \
                read(RamLocation.ITEM_CURSOR_X) == 15 and \
                read(RamLocation.ITEM_CURSOR_Y) == 8 and \
                read(RamLocation.TILE_MAP_BASE + 14 * 20 + 8) == CHARMAP["n"] and \
                read(RamLocation.TILE_MAP_BASE + 14 * 20 + 9) == CHARMAP["i"] and \
                read(RamLocation.TILE_MAP_BASE + 14 * 20 + 10) == CHARMAP["c"]:
            return BattleState.NICKNAME
    elif textbox_id == 0x0b and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 16) == CHARMAP["<PK>"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 17) == CHARMAP["<MN>"]:
        # battle menu
        # if self.print_debug: print(f'is in battle menu at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id in [0x0b, 0x01] and \
            read(RamLocation.TILE_MAP_BASE + 17 * 20 + 4) == CHARMAP["└"] and \
            read(RamLocation.TILE_MAP_BASE + 8 * 20 + 10) == CHARMAP["┐"] and \
            read(RamLocation.ITEM_CURSOR_X) == 5 and \
            read(RamLocation.ITEM_CURSOR_Y) == 12:
        # fight submenu
        # if self.print_debug: print(f'is in fight submenu at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id == 0x0d and \
            read(RamLocation.TILE_MAP_BASE + 2 * 20 + 4) == CHARMAP["┌"] and \
            read(RamLocation.ITEM_CURSOR_X) == 5 and \
            read(RamLocation.ITEM_CURSOR_Y) == 4:
        # bag submenu
        # if self.print_debug: print(f'is in bag submenu at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id == 0x01 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 1) == CHARMAP["C"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 2) == CHARMAP["h"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 3) == CHARMAP["o"]:
        # choose pokemon
        # if self.print_debug: print(f'is in choose pokemon at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id == 0x01 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 1) == CHARMAP["B"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 2) == CHARMAP["r"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 3) == CHARMAP["i"]:
        # choose pokemon after opponent fainted
        # choose pokemon after party pokemon fainted
        # if self.print_debug: print(f'is in choose pokemon after opponent fainted at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id == 0x01 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 1) == CHARMAP["U"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 2) == CHARMAP["s"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 3) == CHARMAP["e"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 8) == CHARMAP["?"]:
        # use item in party submenu
        # if self.print_debug: print(f'is in use item in party submenu at step {self.step_count}')
        return BattleState.ACTIONABLE
    elif textbox_id == 0x0c and \
            read(RamLocation.TILE_MAP_BASE + 12 * 20 + 13) == CHARMAP["S"] and \
            read(RamLocation.TILE_MAP_BASE + 12 * 20 + 14) == CHARMAP["W"]:
        # switch pokemon
        return BattleState.SWITCH
    elif textbox_id == 0x14 and \
            read(RamLocation.ITEM_CURSOR_X) == 1 and \
            read(RamLocation.ITEM_CURSOR_Y) == 8 and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 1) == CHARMAP["c"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 2) == CHARMAP["h"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 15) == CHARMAP["?"]:
        # change pokemon yes no menu
        return BattleState.ACTIONABLE
    elif textbox_id == 0x14 and \
            read(RamLocation.ITEM_CURSOR_X) == 15 and \
            read(RamLocation.ITEM_CURSOR_Y) == 8 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 9) == CHARMAP["m"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 10) == CHARMAP["a"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 11) == CHARMAP["k"]:
        # make room for new move
        return BattleState.LEARN_MOVE
    elif textbox_id == 0x01 and \
            read(RamLocation.ITEM_CURSOR_X) == 5 and \
            read(RamLocation.ITEM_CURSOR_Y) == 8 and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 10) == CHARMAP["t"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 11) == CHARMAP["e"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 12) == CHARMAP["n"] and \
            read(RamLocation.TILE_MAP_BASE + 16 * 20 + 13) == CHARMAP["?"] and \
            read(RamLocation.MAX_MENU_ITEM) == 3:
        # choose move to replace
        return BattleState.REPLACE_MOVE
    elif textbox_id == 0x14 and \
            read(RamLocation.ITEM_CURSOR_X) == 15 and \
            read(RamLocation.ITEM_CURSOR_Y) == 8 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 1) == CHARMAP["A"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 2) == CHARMAP["b"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 3) == CHARMAP["a"]:
        # do not learn move
        return BattleState.ABANDON_MOVE
    elif textbox_id == 0x14 and \
            read(RamLocation.ITEM_CURSOR_X) == 15 and \
            read(RamLocation.ITEM_CURSOR_Y) == 8 and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 8) == CHARMAP["n"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 9) == CHARMAP["i"] and \
            read(RamLocation.TILE_MAP_BASE + 14 * 20 + 10) == CHARMAP["c"]:
        # nickname for caught pokemon
        return BattleState.NICKNAME

    return BattleState.OTHER