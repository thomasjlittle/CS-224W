import argparse
import sys
import os

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Add the parent directory to sys.path
sys.path.append(parent_directory)

from graphgym.utils.agg_runs import agg_batch


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--dir", dest="dir", help="Dir for batch of results", required=True, type=str
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        help="metric to select best epoch",
        required=False,
        type=str,
        default="auto",
    )
    return parser.parse_args()


args = parse_args()
agg_batch(args.dir, args.metric)
