from collections import defaultdict
from typing import Tuple

import numpy as np
from ml_collections import ConfigDict
import hashlib
from enum import Enum

def hash_function(
        components: Tuple
):
    """
    Hash a tuple of (Enum, np.ndarray) using hashlib.
    """
    combined_serialised = b''
    for component in components:
        if isinstance(component, int):
            combined_serialised = combined_serialised + b'|' + component.to_bytes(8, "little")
        elif isinstance(component, str):
            combined_serialised = combined_serialised + b'|' + component.encode('utf-8')
        elif isinstance(component, Enum):
            combined_serialised = combined_serialised + b'|' + component.name.encode('utf-8')
        elif isinstance(component, np.ndarray):
            combined_serialised = combined_serialised + b'|' + component.tobytes()

    hasher = hashlib.sha256(usedforsecurity=False)
    hasher.update(combined_serialised)
    return hasher.hexdigest()

class VisitationCounts:

    def __init__(
            self,
            config: ConfigDict
    ):
        """
        Keeps track of global counts for count-based exploration.
        """

        self.decay_power = config.count_based_decay_power

        self.counts = defaultdict(int)

        self.inital_scale = config.count_based_initial_scale


    def push_samples(
            self,
            visited_hashes: dict
    ):
        for h, c in visited_hashes.items():
            self.counts[h] += c

    def get_scales(self) -> "HashCounts":
        return HashCounts(counts=self.counts, inital_scale=self.inital_scale, decay_power=self.decay_power)

class HashCounts:
    def __init__(
            self,
            counts: dict,
            inital_scale: int = 1,
            decay_power: int = 1,
    ):
        """
        Convenience class to return a default scale if the hash is not in the registered hashes yet.
        :param counts: counts for each registered hash
        :param inital_scale: initial scale for unregistered hashes
        :param decay_power: speed at which count rewards decrease
        """
        self.counts = counts
        self.inital_scale = inital_scale
        self.decay_power = decay_power

    def visit(self, hash):
        self.counts[hash] += 1

    def __getitem__(self, item):
        if item not in self.counts:
            return self.inital_scale
        return 1 / (self.counts[item] ** self.decay_power)
