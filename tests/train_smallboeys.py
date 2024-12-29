import wandb

from sacred import Experiment
from ml_collections import ConfigDict

from deepred.polaris_env.pokemon_red.enums import BagItem
from deepred.polaris_env.polaris_red import PolarisRed
from deepred.polaris_utils.callbacks import Callbacks

exp_name = 'smallboeys'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():
    game_path = ''

    if game_path == '':
        raise ValueError("Need a path for the game rom.")

    env_config = dict(
        game_path = game_path,
        episode_length = 1024 * 1000 * 30, # 2048 is short, for debugging purposes.
        enable_start = False,
        enable_roll_party = True,
        enable_pass = False,
        downscaled_screen_shape = (72, 80),
        framestack = 3,
        stack_oldest_only = False, # maybe True could speed up.
        map_history_length = 15,
        flag_history_length = 15,
        enabled_patches = ("out_of_cash_safari", "infinite_time_safari", "instantaneous_text", "nerf_spinners",
                         "victory_road", "elevator", "freshwater_trade", "seafoam_island"),
        checkpoint_identifiers = ("visited_pokemon_centers_count", "badge_count"),
        max_num_checkpoints = 15,
        env_checkpoint_scoring = {"total_rewards": 1},
        default_savestate = "faster_red_post_parcel_pokeballs.state",
        reward_scales = dict(seen_pokemons=5, experience=5, badges=100, events=20, opponent_level=1,
                             blackout=0.5, exploration=0.25, early_termination=10, heal=300, visited_maps=0),
        laziness_delta_t = 2048*5,
        laziness_threshold = 10,
        session_path = "red_tests",
        record = False,
        speed_limit = 2,
        record_skipped_frame = False,
        render = False,
        stream = True,
        bot_name = "deepred"
    )

    env = PolarisRed.env_id

    num_workers = 126 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'deepred.models.small_boeys'
    policy_class = 'PPO'
    model_class = 'SmallBoeysModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 512
    max_seq_len = trajectory_length # if we use RNNs, this should be set to something like 16 or 32. (we should not need rnns)
    train_batch_size = 2048 * num_workers
    n_epochs=1
    minibatch_size = 512*6 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.
    max_queue_size = train_batch_size * 10

    # count-based exploration
    # Our count-based exploration is a bit different, as we only count once a (map, event-flags) per episode,
    # thus, we do not count the total visitation, but more of a number of episodes where this was visited.
    count_based_decay_power = 1/2 # this is commonly used in the literature
    count_based_initial_scale = 1 # base bonus for new entries.
    count_based_discount = 0.9

    # env checkpoint config
    env_checkpoint_temperature = 30 # temperature for the softmax distribution of checkpoints.
    env_checkpoint_score_lr = 0.1 # speed at which we update the scores for the checkpoints
    env_checkpoint_epsilon = 0.2 # frequency at which we pick random checkpoints

    default_policy_config = {

        'discount': 0.998,  # rewards are x0,129 after 2048 steps.
        'gae_lambda': 0.95, # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 5e-3, # encourages exploration
        'lr': 3e-4, #5e-4

        'grad_clip': 0.5,
        'ppo_clip': 0.2, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.5,
        'initial_kl_coeff': 0.5,
        'kl_target': 1.,
        "vf_clip": 10.
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
        checkpoint_frequency=20,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )
    episode_callback_class = Callbacks

    restore = False

    name = "test_smallboeys_checkpoints"

@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
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

    trainer = SynchronousTrainer(config, restore=config["restore"], with_spectator=config["env_config"]["render"])
    trainer.run()
