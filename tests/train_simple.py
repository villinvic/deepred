import wandb

from sacred import Experiment
from ml_collections import ConfigDict

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_utils.callbacks import Callbacks

exp_name = 'train_simple'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():
    game_path = ''

    if game_path == '':
        raise ValueError("Need a path for the game rom.")

    env_config = dict(
        game_path = game_path,
        episode_length = 256,# 1024 * 1000 * 30, # 2048 is short, for debugging purposes.
        enable_start = False,
        enable_roll_party = True,
        enable_pass = False,
        downscaled_screen_shape = (36, 40),
        framestack = 3,
        stack_oldest_only = False, # maybe True could speed up.
        map_history_length = 10,
        flag_history_length = 10,
        enabled_patches = ("out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
                         "victory_road", "elevator", "freshwater_trade", "seafoam_island"),
        reward_scales = dict(seen_pokemons=0, experience=0, badges=0, events=0,  exploration=1),
        reward_laziness_check_freq = 2048*4,
        reward_laziness_limit = 2048*2,
        savestate = "faster_red_post_parcel_pokeballs.state",
        session_path = "red_tests",
        record = False,
        speed_limit = 1,
        record_skipped_frame = False,
        render = False
    )

    env = PolarisRed.env_id

    num_workers = 64 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.simple'
    policy_class = 'PPO'
    model_class = 'SimpleModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 256
    max_seq_len = 256 # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = 256 * num_workers
    n_epochs=3
    minibatch_size = 2048 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 0.5 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.

    default_policy_config = {

        'discount': 0.999,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 0.95, # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 1e-3, # encourages exploration
        'lr': 5e-4, #5e-4

        'grad_clip': 0.5,
        'ppo_clip': 0.2, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.5,
        'initial_kl_coeff': 0.,
        "vf_clip": 0.5
        }

    policy_params = [{
        "name": "deepred_agent",
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
    from deepred.trainer import SynchronousTrainer

    config = ConfigDict(_config)
    PolarisRed(**config["env_config"]).register()

    wandb.init(
        config=_config,
        project="deepred",
        mode='online',
        group="debug",
        name="train_simple",
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = SynchronousTrainer(config, restore=config["restore"], with_spectator=config["env_config"]["render"])
    trainer.run()