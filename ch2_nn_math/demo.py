"""Script for demo-ing MLP."""

import argparse
from distutils.util import strtobool

import numpy as np

from mlp import MLP  # nopep8
from ops import MeanSquaredError, BinaryCrossEntropy, Linear, ReLU, Sigmoid  # nopep8


def main():

    # Define CLI
    parser = cli(description='script for testing MLP from scratch.')

    args = parser.parse_args()

    # Debug
    if args.debug:
        print('Begin debugging...')
        breakpoint()

    # Set random seed
    np.random.seed(args.random_seed)

    # Generate random data just in case
    x = np.random.normal(size=(args.m_examples, args.n_features))

    # Determine task
    if args.task == 'random-regression':
        y = np.random.normal(size=(args.m_examples, args.t_targets))

        model = MLP(
            input_dims=args.n_features,
            hidden_units=args.num_hidden_units,
            targets=args.t_targets,
            learning_rate=args.learning_rate,
            loss_function=MeanSquaredError(),
            hidden_activation=ReLU(),
            target_activation=Linear(),
            l_layers=args.num_layers,
            debug=args.debug)

    elif args.task == 'random-classification':
        y = np.random.choice(a=args.c_categories, size=(
            args.m_examples,), replace=True)

        model = MLP(
            input_dims=args.n_features,
            hidden_units=args.num_hidden_units,
            targets=args.c_categories,
            learning_rate=args.learning_rate,
            loss_function=BinaryCrossEntropy(),
            hidden_activation=ReLU(),
            target_activation=Sigmoid(),
            l_layers=args.num_layers,
            debug=args.debug)

    elif args.task == 'boston-housing-regression':
        raise NotImplementedError

    elif args.task == 'iris-classification':
        raise NotImplementedError

    # Traing the model
    model.fit(x, y, batch_size=args.batch_size, epochs=args.num_epochs)


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

    parser.add_argument(
        '--debug',
        choices=[True, False],
        help='bool to debug',
        type=lambda x: bool(strtobool(x)),
        default=False)

    parser.add_argument(
        '--random-seed',
        help='random seed for reproducibility. (default: 0)',
        type=int,
        default=0)

    random_data = parser.add_argument_group(
        'random-data-params',
        'parameters for random dataset tasks')

    random_data.add_argument(
        '--m-examples',
        help='number of training examples. (default: 32)',
        type=int,
        default=16)

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
        help='number of hidden units in hidden layers. (default: 2)',
        type=int,
        default=2)

    hparams.add_argument(
        '--batch-size',
        help='batch size for training. (default: 4)',
        type=int,
        default=4)

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
