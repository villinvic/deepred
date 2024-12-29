import fcntl

import numpy as np
import pickle
from pathlib import Path

from deepred.polaris_env.additional_memory import AdditionalMemory


def get_checkpoint_ages(
        path: Path,
):
    """
    :param path: path of checkpoints.
    :return: checkpoint ages.
    """
    ckpt_ages = {}

    for ckpt_path in path.iterdir():
        if ckpt_path.is_file():
            mod_time = ckpt_path.stat().st_mtime
            ckpt_ages[ckpt_path.name] = - mod_time
    return ckpt_ages

def pick_random_ckeckpoint(
        path: Path,
):
    ckpts = list(get_checkpoint_ages(path).keys())
    return np.random.choice(ckpts)

def roll_checkpoints(
        path: Path,
        max_checkpoints: int
):
    """
    :param path: path of checkpoints.
    :param max_checkpoints: maximum number of checkpoints to keep.
    :return: the checkpoint to use.
    """
    ckpt_ages = get_checkpoint_ages(path)
    num_ckpt = len(ckpt_ages)
    if num_ckpt < max_checkpoints:
        return (path / f"checkpoint_{num_ckpt}").with_suffix(".ckpt")

    oldest_ckpt = max(ckpt_ages, key=lambda k: ckpt_ages[k])
    oldest_ckpt_path = path / oldest_ckpt
    oldest_ckpt_path.unlink()
    return path / oldest_ckpt


class EnvCheckpoint:
    """
    A class representing a checkpoint in Pokémon Red.
    """

    def __init__(
            self,
            additional_memory: AdditionalMemory | None = None,
            frame: int | None = None,
            step: int | None = None,
            savestate: bytes | None = None,
            ckpt_id: str | None = None,
    ):
        """
        We need to save everything beside the savestate that cannot be inferred from the savestate.
        """
        self.additional_memory = additional_memory
        self.frame = frame
        self.step = step
        self.savestate = savestate
        self.ckpt_id = ckpt_id
        self.loaded = False

    def save(
            self,
            path: Path,
            max_num_checkpoints: int = 15,
    ):
        """
        Save the checkpoint to a file.

        :param path: Path to the folder where the checkpoint will be saved.
        :param max_num_checkpoints: Maximum number of checkpoints with the same checkpoint id.
        """
        path = path / self.ckpt_id
        path.mkdir(exist_ok=True)
        with open("ckpt.lock", "w+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                path = roll_checkpoints(path, max_num_checkpoints)
                data = {
                    'additional_memory': self.additional_memory,
                    'savestate': self.savestate,
                    "frame": self.frame,
                    "step": self.step,
                    "ckpt_id": self.ckpt_id
                }
                with open(path, 'wb') as file:
                    pickle.dump(data, file)
                print(f"created checkpoint: {path}.")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


    def load(
            self,
            path: Path,
            ckpt_id: str
    ):
        """
        Save the checkpoint to a file.
        :param path: Path to the folder where the checkpoint was saved.
        :param ckpt_id: id of the checkpoint (uniform sample over checkpoints with similar ID).
        """
        path = path / ckpt_id
        path = path / pick_random_ckeckpoint(path)

        with open("ckpt.lock", "w+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                with open(path, 'rb') as file:
                    data = pickle.load(file)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        self.additional_memory = data['additional_memory']
        self.savestate = data['savestate']
        self.frame = data['frame']
        self.step = data['step']
        self.ckpt_id = data['ckpt_id']
        self.loaded = True