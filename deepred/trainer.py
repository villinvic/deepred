import importlib
import queue
import threading
import time
from collections import defaultdict
from typing import Dict, Union

import numpy as np
import psutil
import tree
from ml_collections import ConfigDict

from polaris.checkpointing.checkpointable import Checkpointable
from polaris.experience.episode import EpisodeMetrics, NamedPolicyMetrics, PolicyMetrics
from polaris.experience.worker_set import SyncWorkerSet
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams, ParamsMap
from polaris.experience.matchmaking import RandomMatchmaking
from polaris.experience.sampling import ExperienceQueue, SampleBatch
from polaris.utils.metrics import MetricBank, GlobalCounter, GlobalTimer
from ray.util.client import ray

from deepred.polaris_utils.counting import VisitationCounts


class SynchronousTrainer(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
            restore: Union[bool, str] = False,
            with_spectator: bool = True,

    ):
        """
        The SynchronousTrainer class is an example of a trainer that can be used to train synchronous methods, such
        as A2C, PPO, etc.
        The trainer, in the training_step() method, gathers synchronously samples. Any samples that are gathered
        but not up to date with their respective policy (same version) are thrown away.
        This ensures that any data used for policy updates are always on-policy.

        :param config: Trainer config.
        :param restore: If True, restores the trainer from latest checkpoint.
            If a checkpoint path is given, the Trainer will restore from that checkpoint instead.
        :param with_spectator: If True, the worker with ID 0 will become asynchronous and won't be blocked by
            the training process (typically used when the worker is rendering the environment).
        """

        self.config = config
        self.worker_set = SyncWorkerSet(
            config,
            with_spectator=with_spectator
        )

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

        # We can extend to multiple policy types if really needed, but won't be memory efficient
        policy_params = [
            PolicyParams(**ConfigDict(pi_params)) for pi_params in self.config.policy_params
        ]

        PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        self.policy_map: Dict[str, Policy] = {
            policy_param.name: PolicylCls(
                name=policy_param.name,
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                config=self.config,
                policy_config=policy_param.config,
                options=policy_param.options,
                # For any algo that needs to track either we have the online model
                is_online=True,
            )
            for policy_param in policy_params
        }

        self.visitation_counts: Dict[str, VisitationCounts] = {
            pid: VisitationCounts(self.config)
            for pid in self.policy_map
        }
        for pid, policy in self.policy_map.items():
            policy.options["hash_counts"] = self.visitation_counts[pid].get_counts()


        self.params_map = ParamsMap(**{
            name: p.get_params() for name, p in self.policy_map.items()
        })

        self.experience_queue: Dict[str, ExperienceQueue] = {
            pid: ExperienceQueue(self.config)
            for pid in self.policy_map
        }

        self.matchmaking = RandomMatchmaking(
            agent_ids=self.env.get_agent_ids(),
        )

        self.running_jobs = []

        self.metricbank = MetricBank(
            report_freq=self.config.report_freq
        )
        self.metrics = self.metricbank.metrics

        super().__init__(
            checkpoint_config = config.checkpoint_config,

            components={
                "matchmaking": self.matchmaking,
                "config": self.config,
                "params_map": self.params_map,
                "metrics": self.metrics,
                "visitation_counts": self.visitation_counts
            }
        )

        if restore:
            if isinstance(restore, str):
                self.restore(restore_path=restore)
            else:
                self.restore()

            # override with user specified config if needed:
            self.config = config

            # Need to pass the restored references afterward
            self.metricbank.metrics = self.metrics

            env_step_counter = "counters/" + GlobalCounter.ENV_STEPS
            if env_step_counter in self.metrics:
                GlobalCounter[GlobalCounter.ENV_STEPS] = self.metrics["counters/" + GlobalCounter.ENV_STEPS].get()

            for pid, policy in self.policy_map.items():
                policy.setup(self.params_map[pid])

    def training_step(self):
        """
        Executes one step of the trainer:
        1. Sends jobs to available environment workers.
        2. Gathers ready experience and metrics.
        3. Updates policies whose batches are ready.
        4. Reports experience and training metrics to the bank.
        """

        t = []
        t.append(time.time())
        GlobalTimer[GlobalTimer.PREV_ITERATION] = time.time()
        iteration_dt = GlobalTimer.dt(GlobalTimer.PREV_ITERATION)

        t.append(time.time())
        n_jobs = self.worker_set.get_num_worker_available()
        jobs = [self.matchmaking.next(self.params_map) for _ in range(n_jobs)]
        t.append(time.time())

        self.running_jobs += self.worker_set.push_jobs(self.params_map, jobs)
        experience_metrics = []
        t.append(time.time())
        frames = 0
        env_steps = 0

        experience = []
        while len(self.running_jobs) > 0:
            exp, self.running_jobs = self.worker_set.wait(self.params_map, self.running_jobs, timeout=120)
            experience.extend(exp)
            print(f"Collected {len(experience)} experiences.", f"{len(self.running_jobs)} jobs remaining")
            if len(experience)>0:
                for w in self.worker_set.workers:
                    if w not in self.worker_set.available_workers:
                        print(f"Worker still running:{w}....")


        enqueue_time_start = time.time()
        num_batch = 0

        to_push = defaultdict(list)

        for i, exp_batch in enumerate(experience):
            if isinstance(exp_batch, EpisodeMetrics):
                experience_metrics.append(exp_batch)
                env_steps += exp_batch.length
                for pid in exp_batch.policy_metrics.keys():
                    visitation_counts = exp_batch.custom_metrics.pop("to_pop/visited_hash", None)
                    if visitation_counts is not None:
                        self.visitation_counts[pid].push_samples(visitation_counts)
                        self.policy_map[pid].options["hash_counts"] \
                            = self.visitation_counts[pid].get_counts()
                    else:
                        print('Has not found any visitation counts in the metrics?')

            else: # Experience batch
                batch_pid = exp_batch.get_owner()
                owner = self.policy_map[batch_pid]

                if (not self.experience_queue[owner.name].is_ready()) and owner.version == \
                        exp_batch[SampleBatch.VERSION][0]:
                    num_batch += 1
                    exp_batch = exp_batch.pad_sequences()
                    exp_batch[SampleBatch.SEQ_LENS] = np.array(exp_batch[SampleBatch.SEQ_LENS])
                    # print(f"rcved {owner} {exp_batch[SampleBatch.SEQ_LENS]}, version {exp_batch[SampleBatch.VERSION][0]}")
                    to_push[batch_pid].append(exp_batch)
                    self.policy_map[owner.name].stats["samples_generated"] += exp_batch.size()
                    GlobalCounter.incr("batch_count")
                    frames += exp_batch.size()
                elif owner.version != exp_batch[SampleBatch.VERSION][0]:
                    # pass
                    # toss the batch...
                    print(owner.name, owner.version, exp_batch[SampleBatch.VERSION][0],
                          self.params_map[owner.name].version)
                else:
                    # toss the batch...
                    pass

        for pid, batches in to_push.items():
            self.experience_queue[pid].push(batches)
        if frames > 0:
            GlobalTimer[GlobalTimer.PREV_FRAMES] = time.time()
            prev_frames_dt = GlobalTimer.dt(GlobalTimer.PREV_FRAMES)
        if num_batch > 0:
            enqueue_time_ms = (time.time() - enqueue_time_start) * 1000.
        else:
            enqueue_time_ms = None

        n_experience_metrics = len(experience_metrics)
        GlobalCounter[GlobalCounter.ENV_STEPS] += env_steps
        if n_experience_metrics > 0:
            GlobalCounter[GlobalCounter.NUM_EPISODES] += n_experience_metrics

        t.append(time.time())
        training_metrics = {}
        for pid, policy_queue in self.experience_queue.items():
            if policy_queue.is_ready():
                pulled_batch = policy_queue.pull(self.config.train_batch_size)
                if np.any(pulled_batch[SampleBatch.VERSION] != self.policy_map[pid].version):
                    print(f"Had older samples in the batch for policy {pid} version {self.policy_map[pid].version}!"
                          f" {pulled_batch[SampleBatch.VERSION]}")
                    # this should not happen ever.
                    raise KeyboardInterrupt
                train_results = self.policy_map[pid].train(pulled_batch)
                training_metrics[f"{pid}"] = train_results
                GlobalCounter.incr(GlobalCounter.STEP)
                self.visitation_counts[pid].discount()
                self.params_map[pid] = self.policy_map[pid].get_params()


        def mean_metric_batch(b):
            return tree.flatten_with_path(tree.map_structure(
                lambda *samples: np.mean(samples),
                *b
            ))

        if len(training_metrics)> 0:
            for pid, policy_training_metrics in training_metrics.items():
                policy_training_metrics = mean_metric_batch([policy_training_metrics])
                self.metricbank.update(policy_training_metrics, prefix=f"training/{pid}/",
                                       smoothing=self.config.training_metrics_smoothing)
        if len(experience_metrics) > 0:
            for metrics in experience_metrics:
                self.metricbank.update(tree.flatten_with_path(metrics), prefix=f"experience/",
                                       smoothing=self.config.episode_metrics_smoothing)

        ram_info = psutil.virtual_memory()

        misc_metrics =  [
                    (f'{pi}_queue_length', queue.size())
                    for pi, queue in self.experience_queue.items()
                ] + [('RAM_percent_usage', ram_info.percent)]
        if frames > 0:
            misc_metrics.append(("FPS", frames / prev_frames_dt))
        if enqueue_time_ms is not None:
            misc_metrics.append(("experience_enqueue_ms", enqueue_time_ms))

        self.metricbank.update(
            misc_metrics
            , prefix="misc/", smoothing=0.9
        )

    def run(self):
        """
        Runs the trainer in a loop, until the stopping condition is met or a C^ is caught.
        """

        try:
            while not self.is_done(self.metricbank):
                self.training_step()
                self.metricbank.report()
                self.checkpoint_if_needed()
        except KeyboardInterrupt:
            print("Caught C^. Terminating...")