import io
from enum import Enum, IntEnum
from pathlib import Path
from typing import Tuple
import re

from tensorflow.python.types.core import Callable

from deepred.polaris_env.env_checkpointing.env_checkpoint import EnvCheckpoint
from deepred.polaris_env.gamestate import GameState
from deepred.polaris_utils.counting import hash_function


def sanitise_filename(filename: str, replacement: str = "") -> str:
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    sanitized = sanitized.rstrip(". ")
    max_length = 64
    return sanitized[:max_length]

class EnvCheckpointer:
    def __init__(
            self,
            output_dir: Path,
            checkpoint_identifiers: Tuple,
            max_num_checkpoints: int = 15,
            initial_checkpoint_id: str | None = None,
    ):
        """
        takes care of generating checkpoints whenever we are reaching new parts of the state (given by the checkpoint
        identifiers).
        :param output_dir: path used by the console.
        :param checkpoint_identifiers: tuple of gamestate attributes, e.g. ("map", "badges")
        :param max_num_checkpoints: maximum number of checkpoints to keep per checkpoint id.
        """
        self.path = output_dir / "checkpoints"
        self.checkpoint_identifiers = checkpoint_identifiers
        self.checkpoints_created_this_episode = {initial_checkpoint_id}
        self.max_num_checkpoints = max_num_checkpoints
        self.path.mkdir(exist_ok=True)


    def get_checkpoint_id(
            self,
            gamestate: GameState
    ):
        name_fields = []
        for identifier in self.checkpoint_identifiers:
            value = getattr(gamestate, identifier)
            if isinstance(value, Enum):
                value = value.name
            name_fields.append(f"{identifier}[{value}]")

        ckpt_id = "_".join(name_fields)
        return sanitise_filename(ckpt_id)

    def do_checkpoint_if_needed(
            self,
            savestate_fn: Callable,
            gamestate: GameState
    ):
        ckpt_id = self.get_checkpoint_id(gamestate)
        if ckpt_id in self.checkpoints_created_this_episode:
            return

        with io.BytesIO() as savestate:
            savestate.seek(0)
            savestate_fn(savestate)
            savestate.seek(0)

            self.checkpoints_created_this_episode.add(ckpt_id)
            EnvCheckpoint(
                gamestate._additional_memory,
                gamestate._frame,
                gamestate.step,
                savestate,
                ckpt_id,
            ).save(
                self.path,
                self.max_num_checkpoints,
            )





