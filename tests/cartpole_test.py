import wandb
from sacred import Experiment
from ml_collections import ConfigDict

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_utils.callbacks import Callbacks
from polaris.environments.example import PolarisCartPole

exp_name = 'cartpole'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)



@ex.config
def cfg():

    env_config = dict()

    env = PolarisCartPole.env_id

    num_workers = 64 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.cartpole'
    policy_class = 'PPO'
    model_class = 'CartPoleModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 32
    max_seq_len = 32 # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = 64 * num_workers
    n_epochs= 6
    minibatch_size = 512 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 0.5 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.

    default_policy_config = {

        'discount': 0.99,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 1., # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 0., # encourages exploration
        'lr': 3e-4, #5e-4

        'grad_clip': 20.,
        'ppo_clip': 0.3, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.5,
        'initial_kl_coeff': 1.,
        'kl_target': 10.,
        "vf_clip": 1000.
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
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)

    config = ConfigDict(_config)
    PolarisCartPole(**config["env_config"]).register()

    wandb.init(
        config=_config,
        project="deepred",
        mode='online',
        group="debug",
        name="cartpole",
        notes=None,
        dir=config["wandb_logdir"]
    )

    from polaris.trainers.sync_trainer import SynchronousTrainer

    trainer = SynchronousTrainer(config, restore=config["restore"], with_spectator=False)
    trainer.run()