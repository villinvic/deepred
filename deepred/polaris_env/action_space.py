import time
from enum import IntEnum

from pyboy.utils import WindowEvent
from gymnasium.spaces import Discrete

import sys
import select


class CustomEvent(IntEnum):
    ROLL_PARTY = 0

class PolarisRedActionSpace:

    def __init__(
            self,
            enable_start: bool = True,
            enable_pass: bool = False,
            enable_roll_party: bool = True,
    ):
        """
        Represents the action space, as the interface between the bot and the GameBoy console.
        :param enable_start: Whether to enable the button start (can speed up learning, but may prevent learning some
        interesting behavior, such as swapping pokemons)
        :param enable_pass: Whether to enable the no-op action (no button is pressed).
        :param enable_roll_party: Whether to enable the automated roll party action.
        """

        actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        if enable_start:
            actions.append(WindowEvent.PRESS_BUTTON_START)
        if enable_pass:
            actions.append(WindowEvent.PASS)
        if enable_swap:
            actions.append(CustomEvent.ROLL_PARTY)

        self.actions = actions
        self.gym_spec = Discrete(len(self.actions))

        self.key_to_action = {
            "[A": 3,    # Up arrow
            "[B": 0,  # Down arrow
            "[D": 1,  # Right arrow
            "[C": 2,  # Left arrow
            "": 4,
            "0": 5,
            "5": 6,
            "6": 7,
            "7": 8,
        }

    def get_event(
            self,
            action: int,
    ) -> WindowEvent:
        """
        Returns the event corresponding to the selected action
        """
        return self.actions[action]

    def human_input(
            self
    ) -> int:
        """
        Reads user input from stdin, and maps it to an action.
        Blocks until enter was pressed.
        """
        inputs = input("input:").split("\x1b")
        print()
        if len(inputs) > 1:
            inputs.pop(0)
        return self.key_to_action.get(inputs[-1], 7)







