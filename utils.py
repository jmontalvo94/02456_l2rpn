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

def cli():
    """ Command-line interface to run the training procedure

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
            training = json.load(open(args.config_file))['TRAINING']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']

    return args, general, training, nn_params