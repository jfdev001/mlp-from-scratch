"""Script for demo-ing MLP."""

import argparse
from mlp import MLP


def main():

    # Define CLI
    parser = cli(description='script for testing MLP from scratch.')

    args = parser.parse_args()

    # Determine task
    if args.task == 'random-regression':
        pass
    elif args.task == 'random-classification':
        pass
    elif args.task == 'boston-housing-regression':
        pass
    elif args.task == 'iris-classification':
        pass


def cli(description: str):
    parser = argparse.ArgumentParser(
        description=description)

    parser.add_argument(
        'task',
        choices=[
            'iris-classification',
            'boston-housing-regression',
            'random-regression',
            'random-classification'],
        help='specify supervised learning task.')

    random_data = parser.add_argument_group(
        'random-data-params',
        'parameters for random dataset tasks')

    random_data.add_argument(
        '--m-examples',
        help='number of training examples. (default: 32)',
        type=int,
        default=32)

    random_data.add_argument(
        '--n-features',
        help='number of features. (default: 1)',
        type=int,
        default=1)

    random_data.add_argument(
        '--t-targets',
        help='number of targets for regression. (default: 1)',
        type=int,
        default=1)

    random_data.add_argument(
        '--c-categories',
        help='number of categories for classification. \
            NOTE: Only supports binary classification. (default: 2)',
        type=int,
        default=2)

    hparams = parser.add_argument_group(
        'hyperparameters',
        'hyperparametesr for MLP.')

    hparams.add_argument(
        '--num-layers',
        help='number of hidden layers. (default: 1)',
        type=int,
        default=1)

    hparams.add_argument(
        '--num-hidden-units',
        help='number of hidden units in hidden layers. (default: 32)',
        type=int,
        default=32)

    hparams.add_argument(
        '--batch-size',
        help='batch size for training. (default: 32)',
        type=int,
        default=32)

    hparams.add_argument(
        '--num-epochs',
        help='number of epochs to fit model. (default: 2)',
        type=int,
        default=2)

    hparams.add_argument(
        '--learning-rate',
        help='learning rate for gradient descent. (default: 1e-2)',
        type=float,
        default=1e-2)

    return parser


if __name__ == '__main__':
    main()
