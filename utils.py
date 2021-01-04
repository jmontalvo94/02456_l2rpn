import argparse
import json
import numpy as np
import torch
from datetime import datetime

def set_seed_everywhere(seed):
    """ Set the seed for numpy, pytorch

        Args:
           seed (int): the seed to set everything to
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def cli_train():
    """ Command-line interface to run the training procedure.

        Returns:
            args: arguments from CLI and JSON config file
    """

    parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file.')

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=None,
        help='config file',
        required=True
    )

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help='experiment name'
    )

    args = parser.parse_args()

    # Extract config file data as dictionaries
    if args.config_file is not None:
        if '.json' in args.config_file:
            general = json.load(open(args.config_file))['GENERAL']
            params = json.load(open(args.config_file))['TRAIN']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']
            obs_params = json.load(open(args.config_file))['OBS_PARAMS']

    return args, general, params, nn_params, obs_params

def cli_test():
    """ Command-line interface to run the testing procedure, mostly with Runner.

        Returns:
            args: arguments from CLI and JSON config file
    """

    parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file.')

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=None,
        help='config file',
        required=True
    )

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help='experiment name'
    )

    args = parser.parse_args()

    # Extract config file data as dictionaries
    if args.config_file is not None:
        if '.json' in args.config_file:
            general = json.load(open(args.config_file))['GENERAL']
            params = json.load(open(args.config_file))['TEST']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']
            obs_params = json.load(open(args.config_file))['OBS_PARAMS']

    return args, general, params, nn_params, obs_params

def obs_mask(env, obs_params):
    """ Masks the observation to exclude variables.

        Args:
           env: grid2op environment
           obs_params: dict with (True/False) flag per variable
    """
    lengths = env.observation_space.shape # number of elements per variable
    mask = np.array([], dtype=bool)
    for i, k in enumerate(obs_params):
        mask = np.append(mask, np.tile(obs_params[k], lengths[i]))
    return mask

def get_max_values(env, mask):
    """ Masked maximum values of each variables, obtained from statistics.

        Args:
           env: grid2op environment
           mask: array with (True/False) flag per variable
    """
    max_all = np.concatenate([
        np.array([1., 1., 1., 1., 1., 1.]), # time variables
        env.gen_pmax, env.gen_pmax, # prod_p, prod_q
        np.array([142.1, 142.1, 22.0, 13.2, 142.1]), # prod_v
        np.array([26.7, 112.3, 63.1, 9.2, 13.3, 33.2, 11.1, 4.4, 7.5, 16.5, 17.8]), # load p
        np.array([18.6, 78.3, 43.4, 6.4, 9.2, 23.4, 7.8, 3.0, 5.2, 11.6, 12.5]), # load q
        np.array([142.1, 142.1, 142.1, 142.1, 22., 22., 22., 22., 22., 22., 22.]), # load v
        np.tile(1., env.n_line*14 + env.dim_topo + env.n_sub + env.n_gen*2)
    ])
    return max_all[mask]

def cli_train_ll():
    """ Command-line interface to run the training procedure.

        Returns:
            args: arguments from CLI and JSON config file
    """

    parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file.')

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=None,
        help='config file',
        required=True
    )

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help='experiment name'
    )

    args = parser.parse_args()

    # Extract config file data as dictionaries
    if args.config_file is not None:
        if '.json' in args.config_file:
            general = json.load(open(args.config_file))['GENERAL']
            params = json.load(open(args.config_file))['PARAMS']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']

    return args, general, params, nn_params