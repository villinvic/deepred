from collections import defaultdict
from typing import Dict, List

from polaris.experience import EpisodeCallbacks, SampleBatch



class Callbacks(
    EpisodeCallbacks
):
    """
    For now we only use this to send the polaris_env metrics to be logged.
    """

    def on_episode_end(
        self,
        agents_to_policies,
        env_metrics,
        metrics,
    ):
        metrics.update(env_metrics)