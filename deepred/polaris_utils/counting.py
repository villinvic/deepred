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
        self.scales = dict()


    def push_samples(
            self,
            visited_hash: set
    ):
        for h in visited_hash:
            self.counts[h] += 1
            self.scales[h] = 1/np.float_power(self.counts[h], self.decay_power)

    def get_scales(self) -> "HashScales":
        return HashScales(scales=self.scales, inital_scale=self.inital_scale)

class HashScales:
    def __init__(
            self,
            scales: dict,
            inital_scale: int = 1,
    ):
        """
        Convenience class to return a default scale if the hash is not in the registered hashes yet.
        :param scales: scales for each registered hash
        :param inital_scale: initial scale for unregistered hashes
        """
        self.scales = scales
        self.inital_scale = inital_scale

    def __getitem__(self, item):
        return self.scales.get(item, self.inital_scale)

