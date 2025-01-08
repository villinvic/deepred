import wandb

from sacred import Experiment
from ml_collections import ConfigDict

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_utils.callbacks import Callbacks

exp_name = 'double_policy'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():
    game_path = ''

    if game_path == '':
        raise ValueError("Need a path for the game rom.")

    env_config = dict(
        game_path = game_path,
        episode_length = 2048 * 30_000, # 2048 is short, for debugging purposes.
        enable_start = False,
        enable_roll_party = True,
        enable_pass = False,
        downscaled_screen_shape = (72, 80),
        framestack = 1,
        stack_oldest_only = False, # maybe True could speed up.
        map_history_length = 15,
        flag_history_length = 45,
        enabled_patches = ("out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
                         "victory_road", "elevator", "freshwater_trade", "seafoam_island"),
        checkpoint_identifiers = ("badge_count",),
        max_num_savestates_per_checkpoint = 50,
        env_checkpoint_scoring = {"badges_collected_from_checkpoint": 100},
        default_savestate = "faster_red_post_parcel_pokeballs.state",
        reward_scales = dict(
            # progress
            badges=30,
            events=1,

            # battling / party building
            experience=1,
            party_building=0,
            opponent_level=0,
            blackout=0.1,
            fainting=0.05,
            battle_staling=2e-4,

            # helping rewards
            heal=0.1,
            shopping=0.5,
            box_usage=0.5,

            # map exploration
            exploration=0.02,
            seen_pokemons=0,

            # misc
            early_termination=0.,
        ),

        laziness_delta_t = 2048*5,
        laziness_threshold = 4,
        session_path = "red_tests",
        record = False,
        record_skipped_frame=False,
        speed_limit = -1,
        render = True,
        stream = True,
        bot_name = "deepred"
    )

    env = PolarisRed.env_id

    num_workers = 128 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.double_policy'
    policy_class = 'PPO'
    model_class = 'DoublePolicy'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 512
    max_seq_len = trajectory_length # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = trajectory_length * num_workers
    n_epochs=3
    minibatch_size = 2048 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 1/2 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.
    count_based_discount = 0.9

    # env checkpoint config
    env_checkpoint_temperature = 100 # temperature for the softmax distribution of checkpoints.
    env_checkpoint_score_lr = 0.1 # speed at which we update the scores for the checkpoints
    min_save_states = 50 # minimum number of savestates before initialsing a checkpoint.
    env_checkpoint_epsilon = 0.2 # frequency at which we pick random checkpoints

    default_policy_config = {

        'discount': 0.999,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 0.95, # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 1.3e-2, # encourages exploration
        'lr': 2e-4, #5e-4

        'grad_clip': 0.5,
        'ppo_clip': 0.2, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.25,
        'initial_kl_coeff': 1.,
        'kl_target': 0.01,
        "vf_clip": 1e-1
        }

    policy_params = [{
        "name": "deepred_agent",
        "config": default_policy_config
    }]


    compute_advantages_on_workers = True
    wandb_logdir = 'wandb_logs'
    report_freq = 1
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.5

    checkpoint_config = dict(
        checkpoint_frequency=20,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )
    episode_callback_class = Callbacks

    restore = False

    name = "test_double_policy"

@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from deepred.trainer import SynchronousTrainer

    config = ConfigDict(_config)


    dummy_env = PolarisRed(**config["env_config"])
    dummy_env.register()
    config.env_checkpoint_path = dummy_env.console._checkpointer.path


    wandb.init(
        config=_config,
        project="deepred",
        mode='online',
        group="debug",
        name=config["name"],
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = SynchronousTrainer(config, restore=config["restore"], with_spectator=False)#config["env_config"]["render"])
    trainer.run()
