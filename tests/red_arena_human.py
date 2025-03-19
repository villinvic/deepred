import wandb
import time
from typing import Tuple
import tensorflow as tf

from sacred import Experiment
from ml_collections import ConfigDict

from deepred.polaris_env.red_arena.polaris_red_arena import PolarisRedArena
from deepred.arena_trainer import SynchronousTrainer
from deepred.polaris_utils.callbacks import Callbacks


exp_name = 'arena'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():
    game_path = "faster_red6.gbc"
    env_config = dict(
        game_path=game_path,
        episode_length=500,
        human_inputs=False,
        downscaled_screen_shape=(72, 80),
        framestack=1,
        stack_oldest_only=False,
        enabled_patches=(
            "out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
            "victory_road", "elevator", "freshwater_trade", "seafoam_island"
        ),
        reward_scales=dict(experience=1, win=1, fainting=1, timeout=-1),
        session_path="red_arena_tests",
        checkpoint_identifiers=("badge_count",),
        speed_limit=1,
        render=True,
        wild_battle_savestate="wild_battle.state",
        trainer_battle_savestate="trainer_battle.state",
        level_mean_bounds=(5, 60),
        party_level_std_max=10,
        opponent_level_std_max=3,
        wild_battle_chance=1.0,
        record=False,
    )

    default_policy_config = {

        'discount': 0.999,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 0.95,
        # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 1.3e-2,  # encourages exploration
        'lr': 2e-4,  # 5e-4

        'grad_clip': 0.5,
        'ppo_clip': 0.2,  # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.25,
        'initial_kl_coeff': 1.,
        'kl_target': 0.01,
        "vf_clip": 1e-1
    }

    policy_params = [{
        "name": "deepred_agent",
        "config": default_policy_config
    }]

    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.arena'
    policy_class = 'PPO'
    model_class = 'ArenaModel'

    env = PolarisRedArena.env_id
    num_workers = 16

    trajectory_length = 512
    max_seq_len = trajectory_length  # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = trajectory_length * num_workers

    report_freq = 1

    checkpoint_config = dict(
        checkpoint_frequency=20,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )

    episode_callback_class = Callbacks

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 1 / 2  # this is commonly used in the literature
    count_based_initial_scale = 1  # base bonus for new entries.
    count_based_discount = 0.9

    # env checkpoint config
    env_checkpoint_temperature = 100  # temperature for the softmax distribution of checkpoints.
    env_checkpoint_score_lr = 0.1  # speed at which we update the scores for the checkpoints
    min_save_states = 50  # minimum number of savestates before initialsing a checkpoint.
    env_checkpoint_epsilon = 0.2  # frequency at which we pick random checkpoints

    compute_advantages_on_workers = True
    wandb_logdir = 'wandb_logs'
    report_freq = 1
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.5

    name = "test_red_arena"


@ex.automain
def main(_config):
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = ConfigDict(_config)

    dummy_env = PolarisRedArena(**config["env_config"])
    dummy_env.register()

    wandb.init(
        config=_config,
        project="deepred",
        mode='online',
        group="debug",
        name=config["name"],
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = SynchronousTrainer(config)
    trainer.run()