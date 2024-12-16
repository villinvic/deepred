from enum import IntEnum
from typing import List

from pyboy.utils import WindowEvent
from gymnasium.spaces import Discrete

from deepred.polaris_env.gamestate import GameState


class CustomEvent(IntEnum):
    ROLL_PARTY = 77
    DUMP_FRAME = 78
    SAVE_STATE = 79


human_input_dict = {
    # up arrow
    "[A": WindowEvent.PRESS_ARROW_UP,
    # down arrow
    "[B": WindowEvent.PRESS_ARROW_DOWN,
    # left arrow
    "[D": WindowEvent.PRESS_ARROW_LEFT,
    # right arrow
    "[C": WindowEvent.PRESS_ARROW_RIGHT,
    # enter
    "": WindowEvent.PRESS_BUTTON_A,
    "a": WindowEvent.PRESS_BUTTON_A,
    "0": WindowEvent.PRESS_BUTTON_B,
    "5": WindowEvent.PRESS_BUTTON_START,
    "p": WindowEvent.PASS,

    "r": CustomEvent.ROLL_PARTY,
    # dumps the agent data on the disk (in run_dir/human_dumps/)
    "d": CustomEvent.DUMP_FRAME,
    # saves the gamestate (in run_dir/human_dumps/)
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

    def allowed_actions(
            self,
            gamestate: GameState
    ) -> List[int]:
        """
        :param gamestate: Gamestate to look into for checking which action are allowed or not.
        :return:
        """
        disallowed = set()
        if gamestate.is_in_battle:
            disallowed.add(CustomEvent.ROLL_PARTY)
        else:
            if gamestate.walkable_map[3, 4] == 0:
                disallowed.add(WindowEvent.PRESS_ARROW_UP)
            if gamestate.walkable_map[4, 3] == 0:
                disallowed.add(WindowEvent.PRESS_ARROW_LEFT)
            if gamestate.walkable_map[5, 4] == 0:
                disallowed.add(WindowEvent.PRESS_ARROW_DOWN)
            if gamestate.walkable_map[4, 5] == 0:
                disallowed.add(WindowEvent.PRESS_ARROW_RIGHT)

        return [1 if we not in disallowed else 0 for we in self.actions]

    def get_event(
            self,
            action: int,
            gamestate: GameState = None
    ) -> WindowEvent:
        """
        Returns the event corresponding to the selected action
        """
        if self.human_inputs:
            return self.human_input(gamestate)

        return self.actions[action]

    def human_input(
            self,
            gamestate: GameState = None
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

        event = human_input_dict.get(self.human_input_queue.pop(0), WindowEvent.PASS)
        # try:
        #     if gamestate is not None and self.allowed_actions(gamestate)[self.actions.index(event)] == 0:
        #         event = WindowEvent.PASS
        # except:
        #     pass
        return event







