import argparse
import importlib
import json
import os
import pickle
import sys
import time

import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPO, PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization)

import rgail.utils as utils
import wandb
from rgail.airl_utils import AIRLCallback, get_discriminator
from rgail.plot_utils import plot_obs_point, get_plot_func
from rgail.exploration import CostShapingCallback


def load_expert_data(expert_path, num_rollouts):
    expert_mean_reward = []
    all_data = {}
    for i in range(num_rollouts):
        with open(os.path.join(expert_path, "files/EXPERT_AIRL/rollouts", "%s.pkl"%str(i)), "rb") as f:
            current_data = pickle.load(f)
            # We expect data in 'airl' scheme
            assert(current_data['save_scheme'] == 'airl')

        if i == 0:
            all_data = current_data
        else:
            for k in ['current_states', 'next_states', 'actions']:
                all_data[k] = np.concatenate([all_data[k], current_data[k]], axis=0)

        expert_mean_reward.append(current_data['rewards'])

    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = all_data['current_states'].shape[0]/num_rollouts

    return all_data, expert_mean_reward


def get_discriminator(config, obs_dim, acs_dim, is_discrete):
    def get_disc_obs_dim():
        if config.num_spurious_features > 0 and not config.use_inverse_dynamics_model:
            return obs_dim + config.num_spurious_features
        elif config.use_inverse_dynamics_model:
            return config.idm_latent_dim
        else:
            return obs_dim
    base_params = {'obs_dim': get_disc_obs_dim(),
                   'acs_dim': acs_dim, 'is_discrete': is_discrete,
                   'batch_size': config.disc_batch_size,
                   'lr_schedule': get_schedule_fn(config.disc_learning_rate)}
    if config.reward_learning_algorithm == 'tpil':
        from rgail.baselines import ThirdPersonDiscriminator
        tpil_params = {'latent_dim': config.tpil_latent_dim,
                       'feature_extractor_layers': config.tpil_feature_extractor_layers,
                       'reward_classifier_layers': config.tpil_reward_classifier_layers,
                       'domain_classifier_layers': config.tpil_domain_classifier_layers}
        base_params.update(tpil_params)
        return ThirdPersonDiscriminator(**base_params)
    elif config.reward_learning_algorithm == 'vail':
        raise NotImplementedError
    elif config.reward_learning_algorithm == 'gail':
        from rgail.airl_utils import GAILDiscriminator
        gail_params = {'hidden_sizes': config.disc_reward_layers}
        base_params.update(gail_params)
        return GAILDiscriminator(**base_params)
    elif config.reward_learning_algorithm == 'airl':
        from rgail.airl_utils import AIRLDiscriminator
        airl_params = {'reward_net_hidden_sizes': config.disc_reward_layers,
                            'value_net_hidden_sizes': config.disc_value_layers,
                            'gamma':config.reward_gamma,
                            'reward_scheme': config.reward_scheme}
        base_params.update(airl_params)
        return AIRLDiscriminator(**base_params)
    else:
        raise NotImplementedError

def airl(config):
    # Create the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=False,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=False,
                                     reward_gamma=config.reward_gamma
                                     )

    # We don't need cost when taking samples
    sampling_env = utils.make_eval_env(env_id=config.train_env_id,
                                       use_cost_wrapper=False,
                                       normalize_obs=not config.dont_normalize_obs)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=False,
                                   normalize_obs=not config.dont_normalize_obs)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Load expert data
    expert_data, expert_mean_reward = load_expert_data(config.expert_path, config.expert_rollouts)
    expert_agent = PPOLagrangian.load(os.path.join(config.expert_path, "files/best_model.zip"))

    # Logger
    airl_logger = logger.HumanOutputFormat(sys.stdout)

    # Initialize GAIL/AIRL and setup its callbac
    discriminator = get_discriminator(config, obs_dim, acs_dim, is_discrete)

    def check_plotting_feasibility():
        if config.train_env_id in ['DD2B-v0', 'DD3B-v0', 'CDD2B-v0', 'CDD3B-v0']:
            return True
        elif 'Point' in config.train_env_id:
            return True
        elif (config.disc_obs_select_dim is not None and
              len(config.disc_obs_select_dim) < 3 and
              config.disc_acs_select_dim is not None and
              config.disc_acs_select_dim[0] == -1):
            return True
        else:
            return False

    plot_disc = check_plotting_feasibility()

    # Prpeare arguments to inverse dynamics model
    idm_keys = [k for k in config.keys() if k[:3] == 'idm']
    idm_kwargs = {k:config[k] for k in idm_keys}
    idm_kwargs['obs_dim'] = obs_dim
    idm_kwargs['num_spurious_features'] = config.num_spurious_features
    idm_kwargs['acs_dim'] = acs_dim
    idm_kwargs['discrete_actions'] = is_discrete
    idm_kwargs['device'] = config.device

    airl_update = AIRLCallback(discriminator,
                               expert_data,
                               save_dir=config.save_dir,
                               plot_discriminator=plot_disc,
                               normalize_reward=config.normalize_disc_reward,
                               num_spurious_features=config.num_spurious_features,
                               use_inverse_dynamics_model=config.use_inverse_dynamics_model,
                               inverse_dynamics_model_kwargs=idm_kwargs)
    all_callbacks = [airl_update]


    # Define and train model
    model = PPO(
                policy=config.policy_name,
                env=train_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.reward_gamma,
                gae_lambda=config.reward_gae_lambda,
                clip_range=config.clip_range,
                clip_range_vf=config.clip_range_reward_vf,
                ent_coef=config.ent_coef,
                vf_coef=config.reward_vf_coef,
                max_grad_norm=config.max_grad_norm,
                use_sde=config.use_sde,
                sde_sample_freq=config.sde_sample_freq,
                target_kl=config.target_kl,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
                policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    # All callbacks
    save_periodically = callbacks.CheckpointCallback(
            config.save_every, os.path.join(config.save_dir, "models"),
            verbose=0
    )
    save_env_stats = utils.SaveEnvStatsCallback(train_env, config.save_dir)
    save_best = callbacks.EvalCallback(
            eval_env, eval_freq=config.eval_every, deterministic=False,
            best_model_save_path=config.save_dir, verbose=0,
            callback_on_new_best=save_env_stats
    )
    plot_func = get_plot_func(config.train_env_id)
    plot_callback = utils.PlotCallback(
            plot_func, train_env_id=config.train_env_id,
            plot_freq=config.plot_every, plot_save_dir=config.save_dir
    )
    log_mean_callback = utils.LogMeanCallback()

    # Organize all callbacks in list
    all_callbacks.extend([save_periodically,
                          save_best,
                          plot_callback,
                          #log_mean_callback,
                          ])

    # Train
    model.learn(total_timesteps=int(config.timesteps),
                callback=all_callbacks)

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

    # Make video of final model
    if not config.wandb_sweep:
        sync_envs_normalization(train_env, eval_env)
        utils.eval_and_make_video(eval_env, model, config.save_dir, "final_policy")

    if config.sync_wandb:
        utils.sync_wandb(config.save_dir, 120)

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # ======================== Ignore this ========================== #
    parser.add_argument("file_to_run", type=str)
    # ========================== Setup ============================== #
    parser.add_argument("--config_file", "-cf", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default="ABC")
    parser.add_argument("--group", "-g", type=str, default=None)
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False)
    parser.add_argument("--sync_wandb", "-sw", action="store_true")
    # ============================ Cost ============================= #
    parser.add_argument("--cost_info_str", "-cis", type=str, default="cost")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3")
    parser.add_argument("--dont_normalize_obs", "-dno", action="store_true")
    parser.add_argument("--dont_normalize_reward", "-dnr", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    # ======================== Networks ============================== #
    parser.add_argument("--policy_name", "-pn", type=str, default="MlpPolicy")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*')
    parser.add_argument("--policy_layers", "-pl", type=int, default=[64,64], nargs='*')
    parser.add_argument("--reward_vf_layers", "-rl", type=int, default=[64,64], nargs='*')
    # ========================= Training ============================ #
    parser.add_argument("--timesteps", "-t", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--n_steps", "-ns", type=int, default=2048)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--num_threads", "-nt", type=int, default=5)
    parser.add_argument("--save_every", "-se", type=float, default=5e5)
    parser.add_argument("--eval_every", "-ee", type=float, default=2048)
    parser.add_argument("--plot_every", "-pe", type=float, default=2048)
    # =========================== MDP =============================== #
    parser.add_argument("--reward_gamma", "-rg", type=float, default=0.99)
    parser.add_argument("--reward_gae_lambda", "-rgl", type=float, default=0.95)
    # ========================= Losses ============================== #
    parser.add_argument("--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--clip_range_reward_vf", "-crv", type=float, default=None)
    parser.add_argument("--ent_coef", "-ec", type=float, default=0.)
    parser.add_argument("--reward_vf_coef", "-rvc", type=float, default=0.5)
    parser.add_argument("--target_kl", "-tk", type=float, default=None)
    parser.add_argument("--max_grad_norm", "-mgn", type=float, default=0.5)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    # =========================== SDE =============================== #
    parser.add_argument("--use_sde", "-us", action="store_true")
    parser.add_argument("--sde_sample_freq", "-ssf", type=int, default=-1)
    # ================ Algorithm For Reward Learning ================ #
    parser.add_argument("--reward_learning_algorithm", "-algo", type=str, default="gail")
    # =================== AIRL Discriminator ======================== #
    parser.add_argument("--reward_scheme", "-rs", type=str, default="sa",
    help="reward scheme to be used by AIRL discriminator; must be in ['s', 'ss', 'sa']")
    parser.add_argument("--normalize_disc_reward", "-ndr", action="store_true")
    parser.add_argument("--disc_reward_layers", "-drl", type=int, default=[30,30], nargs='*',
    help="If gail is used then this controls gail discriminator capacity.")
    parser.add_argument("--disc_value_layers", "-dvl", type=int, default=[30,30], nargs='*')
    parser.add_argument("--disc_learning_rate", "-dlr", type=float, default=3e-4)
    parser.add_argument("--disc_batch_size", "-dbs", type=int, default=None)
    parser.add_argument('--disc_obs_select_dim', '-dosd', type=int, default=None, nargs='+')
    parser.add_argument('--disc_acs_select_dim', '-dasd', type=int, default=None, nargs='+')
    parser.add_argument('--disc_plot_every', '-dpe', type=int, default=1)
    parser.add_argument('--disc_normalize', '-cn', action='store_true')
    parser.add_argument("--disc_eps", "-de", type=float, default=1e-5)
    parser.add_argument("--update_freq", "-uf", type=int, default=1)
    # ================= Inverse Dynamics Model ======================= #
    parser.add_argument("--use_inverse_dynamics_model", "-uidm", action="store_true")
    parser.add_argument('--idm_loss_type', '-idmlt', type=str, default='fi')
    parser.add_argument("--idm_latent_dim", "-idmld", type=int, default=10)
    parser.add_argument("--idm_encoder_layers", "-idmel", type=int, default=[32,32], nargs='*')
    parser.add_argument("--idm_inverse_model_layers", "-idmiml", type=int, default=[32,32], nargs='*')
    parser.add_argument("--idm_forward_model_layers", "-idmfml", type=int, default=[32,32], nargs='*')
    parser.add_argument("--idm_learning_rate", "-idmlr", type=float, default=3e-4)
    parser.add_argument("--idm_batch_size", "-idmbs", type=int, default=None)
    # ======================== Expert Data ========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/HCWithPos-New')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=20)
    # ======================= Spurious Features ==================== #
    parser.add_argument("--num_spurious_features", "-nsf", type=int, default=0)
    # =============================================================== #
    # ======================= Baselines ============================= #
    # =============================================================== #
    # ============ Third Person Imitation Learning ================== #
    parser.add_argument("--tpil_latent_dim", "-tpilld", type=int, default=10)
    parser.add_argument("--tpil_feature_extractor_layers", "-tpilfl", type=int, default=[32,32], nargs='*')
    parser.add_argument("--tpil_domain_classifier_layers", "-tpildl", type=int, default=[32,32], nargs='*')
    parser.add_argument("--tpil_reward_classifier_layers", "-tpilrl", type=int, default=[32,32], nargs='*')
    # ======================== VAIL ================================= #

    # =============================================================== #

    args = vars(parser.parse_args())

    # Get default config
    default_config, mod_name = {}, ''
    if args["config_file"] is not None:
        if args["config_file"].endswith(".py"):
            mod_name = args["config_file"].replace('/', '.').strip(".py")
            default_config = importlib.import_module(mod_name).config
        elif args["config_file"].endswith(".json"):
            default_config = utils.load_dict_from_json(args["config_file"])
        else:
            raise ValueError("Invalid type of config file")

    # Overwrite config file with parameters supplied through parser
    # Order of priority: supplied through command line > specified in config
    # file > default values in parser
    config = utils.merge_configs(default_config, parser, sys.argv[1:])

    # Choose seed
    if config["seed"] is None:
        config["seed"] = np.random.randint(0,100)

    # Get name by concatenating arguments with non-default values. Default
    # values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config["name"] = utils.get_name(parser, default_config, config, mod_name)

    # Initialize W&B project
    wandb.init(project=config["project"], name=config["name"], config=config, dir="./rgail",
               group=config['group'])
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
          color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    airl(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f minutes" % ((end-start)/60),
          color="green", bold=True))
