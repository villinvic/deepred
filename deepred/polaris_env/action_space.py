from enum import IntEnum

from pyboy.utils import WindowEvent
from gymnasium.spaces import Discrete


class CustomEvent(IntEnum):
    ROLL_PARTY = 77
    DUMP_FRAME = 78
    SAVE_STATE = 79


human_input_dict = {
    "[A": WindowEvent.PRESS_ARROW_UP,
    "[B": WindowEvent.PRESS_ARROW_DOWN,
    "[D": WindowEvent.PRESS_ARROW_LEFT,
    "[C": WindowEvent.PRESS_ARROW_RIGHT,
    "": WindowEvent.PRESS_BUTTON_A,
    "a": WindowEvent.PRESS_BUTTON_A,

    "0": WindowEvent.PRESS_BUTTON_B,
    "5": WindowEvent.PRESS_BUTTON_START,
    "p": WindowEvent.PASS,
    "r": CustomEvent.ROLL_PARTY,
    "d": CustomEvent.DUMP_FRAME,
    "s": CustomEvent.SAVE_STATE,

}


class PolarisRedActionSpace:

    def __init__(
            self,
            enable_start: bool = True,
            enable_pass: bool = False,
            enable_roll_party: bool = True,
            human_inputs: bool = False,
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
        if enable_roll_party:
            actions.append(CustomEvent.ROLL_PARTY)

        self.human_inputs = human_inputs
        self.human_input_queue = []

        self.actions = actions
        self.gym_spec = Discrete(len(self.actions))

    def get_event(
            self,
            action: int,
    ) -> WindowEvent:
        """
        Returns the event corresponding to the selected action
        """
        if self.human_inputs:
            return self.human_input()

        return self.actions[action]

    def human_input(
            self
    ) -> WindowEvent:
        """
        Reads user input from stdin, and maps it to an action.
        Blocks until enter was pressed.
        """
        if len(self.human_input_queue) == 0:
            inputs = input("input:").split("\x1b")
            print()
            if len(inputs) > 1:
                inputs.pop(0)
            self.human_input_queue = inputs
        return human_input_dict.get(self.human_input_queue.pop(0), WindowEvent.PASS)







