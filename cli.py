import argparse
import json
from datetime import datetime

def cli():

    parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file. Optionally override values only from general section of config file, e.g. using "--seed 456" in the CLI will override the default parameter from the JSON file, but using "--activation sigmoid" will not.')

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
            nn_params = json.load(open(args.config_file))['NN_PARAMS']

    return args, general, nn_params

def cli_extended():

    cli_parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file. Optionally override values only from general section of config file, e.g. using "--seed 456" in the CLI will override the default parameter from the JSON file, but using "--activation sigmoid" will not.')

    cli_parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=None,
        help='config file',
        required=True
    )

    cli_parser.add_argument(
        '-n',
        '--name',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help='experiment name'
    )

    args, unknown = cli_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False)

    if args.config_file is not None:
        if '.json' in args.config_file:
            general = json.load(open(args.config_file))['GENERAL']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']
            
            parser.set_defaults(**general)

            [
                parser.add_argument(arg, type=type(general[arg.split('--')[-1]]))
                for arg in [arg for arg in unknown if arg.startswith('--')]
                if arg.split('--')[-1] in general
            ]

    args = parser.parse_args()

    return args, general, nn_params

if __name__ == "__main__":

    args, general, nn_params = cli()
    print(args)
    print(general)
    print(nn_params)