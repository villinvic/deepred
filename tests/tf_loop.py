import gc
import importlib

import numpy as np
import psutil
import tree
import wandb
from polaris.policies import PolicyParams, Policy
from sacred import Experiment
from ml_collections import ConfigDict
from tensorflow.python.keras.backend import dtype

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_utils.callbacks import Callbacks
from polaris.environments.example import PolarisCartPole

exp_name = 'tf_loop'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)



@ex.config
def cfg():

    env_config = dict()

    env = PolarisCartPole.env_id

    num_workers = 16 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.cartpole'
    policy_class = 'PPO'
    model_class = 'CartPoleModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 128
    max_seq_len = 128 # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = 2048 * num_workers
    n_epochs= 3
    minibatch_size = 2048 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 0.5 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.

    default_policy_config = {

        'discount': 0.99,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 1., # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 0.01, # encourages exploration
        'lr': 3e-4, #5e-4

        'grad_clip': 1.,
        'ppo_clip': 0.2, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.01,
        'initial_kl_coeff': 1.,
        'kl_target': 0.05,
        "vf_clip": 10.
        }

    policy_params = [{
        "name": "agent",
        "config": default_policy_config
    }]


    compute_advantages_on_workers = True
    wandb_logdir = 'wandb_logs'
    report_freq = 1
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )

    episode_callback_class = Callbacks

    restore = False

@ex.automain
def main(_config):
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    current_process = psutil.Process(os.getpid())

    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)


    config = ConfigDict(_config)
    env = PolarisCartPole(**config["env_config"])


    policy_param = PolicyParams(**config.policy_params[0])

    PolicylCls = getattr(importlib.import_module(config.policy_path), config.policy_class)
    policy: Policy = PolicylCls(
            name=policy_param.name,
            action_space=env.action_space,
            observation_space=env.observation_space,
            config=config,
            policy_config=ConfigDict(policy_param.config),
            options=policy_param.options,
            # For any algo that needs to track either we have the online model
            is_online=True,
    )

    obs = env.observation_space.sample()
    state = policy.get_initial_state()
    n = 60000
    obs = np.zeros((3, 5) + obs.shape, dtype=np.float32)
    prev_action = np.zeros((3, 5), dtype=np.float32)
    prev_reward = np.zeros((3, 5), dtype=np.float32)
    state = tree.map_structure(
        lambda s: np.zeros((5,) + s.shape, dtype=np.float32),
        state
    )
    seq_lens = np.zeros((5,), dtype=np.float32) * 3

    for i in range(n):
        if i % (n//20) == 0:
            # tf.keras.backend.clear_session()
            # gc.collect()
            # Get memory info
            memory_info = current_process.memory_info()

            # Convert to MB
            ram_usage_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size (RSS) in bytes
            print(f"{i}/{n}...", f"RAM usage (MB): {ram_usage_mb:.2f}")
        policy.compute_single_action(
            obs=obs,
            prev_action=0,
            prev_reward=0.,
            state=state
        )



