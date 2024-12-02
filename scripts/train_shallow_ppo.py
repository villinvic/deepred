import wandb

from sacred import Experiment, Ingredient
from ml_collections import ConfigDict

from deepred.polaris_env.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_env.rewards import Goals
from deepred.polaris_utils.callbacks import Callbacks

exp_name = 'ShallowModel'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():
    game_path = ''

    if game_path == '':
        raise ValueError("Need a path for the game rom.")

    env_config = dict(
        game_path = game_path,
        episode_length = 2048 * 4, # 2048 is short, for debugging purposes.
        enable_start = True,
        enable_pass = False,
        downscaled_screen_shape = (36, 40),
        framestack = 3,
        stack_oldest_only = False, # maybe True could speed up.
        observed_ram = ("badges", "money", "party_hp", "party_level", "sent_out", "in_battle",
                                    #"bag_items",
                                    #"event_flags"
                        ),
        observed_items = (BagItem.POKE_BALL, BagItem.GREAT_BALL, BagItem.ULTRA_BALL, BagItem.MASTER_BALL,
                                          BagItem.POTION, BagItem.SUPER_POTION, BagItem.HYPER_POTION, BagItem.FULL_RESTORE),
        reward_scales = dict(seen_pokemons=0, experience=10, badges=0, exploration=1),
        savestate = "faster_red_post_parcel_pokeballs.state",
        session_path = "red_tests",
        record = False,
        speed_limit = 1,
        record_skipped_frame = False,
        render=False
    )

    env = PolarisRed.env_id

    num_workers = 64 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.impala_shallow'
    policy_class = 'PPO'
    model_class = 'ImpalaShallowModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 512
    max_seq_len = 512 # if we use RNNs, this should be set to something like 16 or 32.
    train_batch_size = env_config["episode_length"] * 8
    n_epochs=1
    minibatch_size = 512 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 0.5 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.

    default_policy_config = {

        'discount': 0.997,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 0.95, # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 0., # encourages exploration
        'lr': 5e-4,

        'grad_clip': 5.,
        'ppo_clip': 0.5, # smaller clip coefficient will lead to more conservative updates.
        'initial_kl_coeff': 0.,
        'baseline_coeff': 0.5,
        'vf_clip': 1.,
        'kl_target': 1e-3, # target KL divergence, a smaller target will lead to more conservative updates.
        'aux_loss_weight': 0.05, # Unused for now.
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
        name="shallow_ppo",
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = SynchronousTrainer(config, restore=config["restore"], with_spectator=config["env_config"]["render"])
    trainer.run()