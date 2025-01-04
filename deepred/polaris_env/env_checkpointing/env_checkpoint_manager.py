from pathlib import Path
from typing import Union, Tuple

import numpy as np
import tree

from deepred.polaris_env.env_checkpointing.env_checkpoint import EnvCheckpoint


class EnvCheckpointManager:

    def __init__(
            self,
            temperature: float,
            score_lr: float,
            min_save_states: int,
            epsilon: float,
            checkpoint_path: Path,
    ):
        """
        Utility class that generates a distribution over checkpoints for the agent.
        :param temperature: annealing factor for the softmax.
        :param score_lr: speed at which the scores are updated (EMA learning rate).
        :param min_save_states: minimum amount of savestates for a given checkpoint before sampling the checkpoint
        :param epsilon: minimum probability of picking any checkpoint.
        :param checkpoint_path: Path to the checkpoint dirs.
        """
        self.temperature = temperature
        self.lr = score_lr
        self.min_save_states = min_save_states
        self.epsilon = epsilon
        self.path = checkpoint_path
        self.scores = {}
        self.distribution = {}
        self.sampled_counts = {}

    def update(self):
        self.read_available_checkpoints()
        if len(self.scores) > 0:
            self.compute_checkpoint_distribution()

    def compute_checkpoint_distribution(self):
        """
        You want to pick checkpoints where the bot performs worse more often
        :return:
        """
        min_s = min(self.scores.values())

        weights = tree.map_structure(
            lambda s: np.exp((min_s - s)/self.temperature),
            self.scores
        )
        z = sum(weights.values())
        n = len(self.scores)
        self.distribution = tree.map_structure(
            lambda w: (w / z) * (1-self.epsilon) + self.epsilon / n,
            weights
        )
        print("Env checkpoint distributions:")
        for ckpt_id, p in self.distribution.items():
            print(f"{ckpt_id[:41]:<40}: {p:.2f}")

    def read_available_checkpoints(self):

        for subdir in self.path.iterdir():
            if subdir.is_file():
                continue
            num_ckpts = sum(1 for file in subdir.iterdir() if file.is_file())
            ckpt_id = subdir.name

            if num_ckpts < self.min_save_states:
                # not enough savestates to initialise the checkpoint
                return

            if ckpt_id not in self.scores:
                # if no checkpoint is currently registered, initialise at a default value (0)
                if len(self.scores) == 0:
                    self.scores[ckpt_id] = 0
                # otherwise initialise at the current min score achieved by other checkpoints. (to introduce the new
                # checkpoint smoothly)
                else:
                    min_score = np.min(list(self.scores.values()))
                    self.scores[ckpt_id] = min_score

    def update_scores(
            self,
            ckpt_id: str,
            score: float
    ):
        assert ckpt_id in self.scores, (ckpt_id, self.scores)

        old_score = self.scores[ckpt_id]
        self.scores[ckpt_id] = (1 - self.lr) * old_score + self.lr * score

        if ckpt_id not in self.sampled_counts:
            self.sampled_counts[ckpt_id] = 0

        self.sampled_counts[ckpt_id] += 1

    def get_sampler(self) -> Union["EnvCheckpointSampler", None]:
        if len(self.scores) == 0:
            return
        return EnvCheckpointSampler(self.path, self.distribution)

    def get_metrics(self):
        if len(self.scores) == 0:
            return {}
        # pass np arrays to interpret these as barplots
        return {
            "checkpoint_scores": np.array(self.scores),
            "checkpoint_distribution": np.array({k: v * 100 for k, v in self.distribution.items()}),
            "checkpoint_sampled_count": np.array(self.sampled_counts)
        }


class EnvCheckpointSampler:

    def __init__(
            self,
            path: Path,
            distribution: dict
    ):
        self.path = path
        self.ckpt_ids = list(distribution.keys())
        self.num_ckpt = len(self.ckpt_ids)
        self.p =  list(distribution.values())

    def __call__(self) -> Tuple[str, int, EnvCheckpoint]:

        ckpt_num = np.random.choice(self.num_ckpt, p=self.p)
        ckpt_id = self.ckpt_ids[ckpt_num]
        checkpoint = EnvCheckpoint()
        checkpoint.load(self.path, ckpt_id)
        print(ckpt_num, ckpt_id)
        return ckpt_id, ckpt_num, checkpoint






