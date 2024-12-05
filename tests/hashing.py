import time

from deepred.polaris_env.pokemon_red.enums import Map
from deepred.polaris_utils.counting import hash_function
import numpy as np


enum = Map.MT_MOON
random = np.random.default_rng(0)
flags = np.uint8(
    random.integers(0, 2, (500,))
)
t = time.time()
print(hash_function((enum, flags)))
print(time.time()-t)
t = time.time()
print(hash_function((enum, flags)))
print(time.time()-t)