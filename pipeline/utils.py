import os
import sys
import argparse
from glob import glob

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const


def get_latest_model(args):
    model_path = const.SAVE_PATH    
    models = glob(f"{model_path}/{const.DATASET}_{const.DATA_SEGMENT}_{const.BACKBONE_NAME}_{const.NUM_JOINTS}_a2j.pth")
    return models[0]


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Pipeline training")

    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="set to True to resume training.",
    )
    
    parser.add_argument(
        "--resume_from_model",
        type=str,
        default=None,
        help="Full path to a model checkpoint.",
    )

    return parser.parse_args()

