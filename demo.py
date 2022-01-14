"""Script for demo-ing MLP."""

import argparse
from distutils.util import strtobool

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError as TFMeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy as TFBinaryCrossEntropy

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
    tf.random.set_seed(args.random_seed)

    # Generate random data just in case
    x = np.random.normal(size=(args.m_examples, args.n_features))

    # Tensorflow model
    baseline_model = Sequential()
    for lyr in range(args.num_layers):
        baseline_model.add(
            Dense(units=args.num_hidden_units, activation='relu'))

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

        baseline_model.add(Dense(units=args.t_targets))

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

        baseline_model.add(
            Dense(units=args.c_categories, activation='sigmoid'))

    elif args.task == 'diabetes-regression':
        x, y = datasets.load_diabetes(return_X_y=True)

        y = np.expand_dims(y, axis=-1)

        model = MLP(
            input_dims=x.shape[-1],
            hidden_units=args.num_hidden_units,
            targets=1,
            learning_rate=args.learning_rate,
            loss_function=MeanSquaredError(),
            hidden_activation=ReLU(),
            target_activation=Linear(),
            l_layers=args.num_layers,
            debug=args.debug)

        baseline_model.add(Dense(units=1))

    elif args.task == 'breast-cancer-classification':
        x, y = datasets.load_breast_cancer(return_X_y=True)

        model = MLP(
            input_dims=x.shape[-1],
            hidden_units=args.num_hidden_units,
            targets=2,
            learning_rate=args.learning_rate,
            loss_function=BinaryCrossEntropy(),
            hidden_activation=ReLU(),
            target_activation=Sigmoid(),
            l_layers=args.num_layers,
            debug=args.debug)

        baseline_model.add(Dense(units=2, activation='sigmoid'))

    # Traing the hand implemented model
    my_history = model.fit(
        x, y, batch_size=args.batch_size, epochs=args.num_epochs,
        test_size=args.test_size, random_state=args.random_seed)

    # Compile reference model
    if 'classification' in args.task:
        baseline_model.compile(
            loss=TFBinaryCrossEntropy(),
            optimizer='sgd')
    else:
        baseline_model.compile(
            loss=TFMeanSquaredError(),
            optimizer='sgd')

    # Train reference model
    tf_history = baseline_model.fit(
        x=x, y=y, batch_size=args.batch_size, epochs=args.num_epochs,
        validation_split=args.test_size)


def cli(description: str):
    parser = argparse.ArgumentParser(
        description=description)

    parser.add_argument(
        'task',
        choices=[
            'breast-cancer-classification',
            'diabetes-regression',
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
        '--test-size',
        help='percent of data to devote to testing. (default: 0.2)',
        type=float,
        default=0.2)

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
