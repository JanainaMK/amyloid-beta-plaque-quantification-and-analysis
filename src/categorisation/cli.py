from argparse import ArgumentParser

import yaml
from pytorch_lightning import seed_everything

def parse_config():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
    )
    args = parser.parse_args()

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    seed_everything(config["exp_params"]["manual_seed"], True)

    return config