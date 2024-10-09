from argparse import ArgumentParser

import yaml
from pytorch_lightning import seed_everything


def add_io_settings(parser: ArgumentParser):
    parser.add_argument(
        "-sf",
        "--source_folder",
        required=True,
        type=str,
        help="the folder in which the vsi file(s) is/are located",
    )
    parser.add_argument(
        "-tf",
        "--target_folder",
        required=True,
        type=str,
        help="The folder in which to place the resulting hdf5 file(s).",
    )


def add_segmentation_settings(parser: ArgumentParser):
    parser.add_argument(
        "-pss",
        "--patch_size_segmentation",
        default=1024,
        type=int,
        help="The patch size in pixels used in the segmentation step (default=1024)",
    )
    parser.add_argument(
        "-df",
        "--downscale_factor",
        default=16,
        type=int,
        help="The downsampling factor used for creating the grey matter segmentation (default=16)",
    )
    parser.add_argument(
        "-mp",
        "--model_path",
        default="models/segmentation/unet/2023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt",
        type=str,
        help="The path to the unet model trained for the segmentation task.",
    )


def add_localisation_settings(parser: ArgumentParser):
    parser.add_argument(
        "-psl",
        "--patch_size_localisation",
        default=4096,
        type=int,
        help="The patch size in pixels used in the localisation step (default=4096)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.04,
        type=float,
        help="The minimum threshold parameter for the localisation step. Range from 0 to 1 (default=0.04)",
    )
    parser.add_argument(
        "-ks",
        "--kernel_size",
        default=21,
        type=int,
        help="The kernel size parameter in pixels for the localisation step (default=21)",
    )
    parser.add_argument(
        "-ms",
        "--minimum_size",
        default=10,
        type=float,
        help="The minimum plaque size parameter in microns for the localisation step (default=10)",
    )


def add_series_settings(parser: ArgumentParser):
    parser.add_argument(
        "-s0",
        "--start",
        default=0,
        type=int,
        help="The index of the file in the source folder where the script should start",
    )
    parser.add_argument(
        "-s1",
        "--stop",
        default=2**32 - 1,
        type=int,
        help="The index of the file in the source folder where the script should stop (exclusive)",
    )


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
