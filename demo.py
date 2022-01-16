"""Script for demo-ing MLP."""

import argparse
from distutils.util import strtobool
import warnings

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError as TFMeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy as TFBinaryCrossEntropy

from mlp import MLP  # nopep8
from ops import MeanSquaredError, BinaryCrossEntropy, Linear, ReLU, Sigmoid  # nopep8
from stats import MultiModelHistory, plot_bar_charts, plot_train_val_loss  # nopep8


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
        y = np.random.choice(a=2, size=(
            args.m_examples,), replace=True)

        model = MLP(
            input_dims=args.n_features,
            hidden_units=args.num_hidden_units,
            targets=2,
            learning_rate=args.learning_rate,
            loss_function=BinaryCrossEntropy(),
            hidden_activation=ReLU(),
            target_activation=Sigmoid(),
            l_layers=args.num_layers,
            debug=args.debug)

        # # Set shape
        # if args.c_categories == 2:
        #     args.c_categories = 1

        baseline_model.add(
            Dense(units=1, activation='sigmoid'))

    elif args.task == 'diabetes-regression':
        x, y = datasets.load_diabetes(return_X_y=True)

        # Scale datasets
        x_scaler = StandardScaler()
        x = x_scaler.fit_transform(x)

        y = np.expand_dims(y, axis=-1)
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y)

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

        # Scale x
        x_scaler = StandardScaler()
        x = x_scaler.fit_transform(x)

        # Targets is 2 to indicate two categories for special case...
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

        # One output unit
        baseline_model.add(Dense(units=1, activation='sigmoid'))

    # Log shapes
    print('x, y shapes:', x.shape, y.shape)

    # Compile reference model
    if 'classification' in args.task:
        baseline_model.compile(
            loss=TFBinaryCrossEntropy(),
            optimizer='sgd')
    else:
        baseline_model.compile(
            loss=TFMeanSquaredError(),
            optimizer='sgd')

    # Cross validation and building of models
    multi_model_history = MultiModelHistory()
    for n in range(args.n_kfold_iterations):

        # Instantiate kfold object
        kfold = KFold(shuffle=True, random_state=n)

        # i^{th} kfold cv
        for train_ix, test_ix in kfold.split(x):

            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            # Training the hand implemented model
            my_history = model.fit(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                test_size=args.test_size,
                random_state=args.random_seed,
                verbose=args.verbose)

            # Train reference model
            tf_history = baseline_model.fit(
                x=x_train, y=y_train,
                validation_data=(x_test, y_test),
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                verbose=args.verbose)

            # Update the multimodel history dictionary
            multi_model_history.append_kth_fold_model_history(
                model_history=my_history, model_key='my_model',)

            multi_model_history.append_kth_fold_model_history(
                model_history=tf_history.history, model_key='tf_model')

    # Can be used to observe whether vanishing or exploding gradient
    # issues occurred
    if args.inspect_model_history:
        print(multi_model_history.nested_dict)
        breakpoint()

    # Check to see if any nan values are contained in the nested
    # dictionaries... this can occur when batching is too large for
    # epochs
    for model_name, metric_key_value_list_dict in multi_model_history.nested_dict.items():
        for metric_key, value_list in metric_key_value_list_dict.items():
            nan_array = np.isnan(value_list)
            if nan_array.any():
                num_nans = np.count_nonzero(nan_array)
                warnings.warn(
                    f'{model_name}: {metric_key} has `{num_nans}` np.nan')

    # Bar chart
    bar_chart = plot_bar_charts(
        multi_model_history=multi_model_history,
        bar_width=args.bar_width,
        title=args.bar_plot_title,
        alpha=args.confidence_level)

    bar_chart.savefig(args.bar_chart_path, bbox_inches='tight')

    # # Loss curves
    # multi_model_history.reshape_metrics_to_nkfolds_by_epochs(
    #     nkfolds=args.n_kfold_iterations*5, epochs=args.num_epochs)
    # loss_curve = None


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
        '--inspect-model-history',
        choices=[True, False],
        help='bool to print model history at the end of N, KFold CV',
        type=lambda x: bool(strtobool(x)),
        default=False)

    parser.add_argument(
        '--random-seed',
        help='random seed for reproducibility. (default: 0)',
        type=int,
        default=0)

    parser.add_argument(
        '--verbose',
        choices=[True, False],
        help='whether to print model fitting output. (default: False)',
        type=lambda x: bool(strtobool(x)),
        default=False)

    random_data = parser.add_argument_group(
        'random-data-params',
        'parameters for random dataset tasks')

    random_data.add_argument(
        '--m-examples',
        help='number of training examples. (default: 32)',
        type=int,
        default=64)

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

    # random_data.add_argument(
    #     '--c-categories',
    #     help='number of categories for classification. \
    #         NOTE: Only supports binary classification. (default: 2)',
    #     type=int,
    #     default=2)

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

    stats = parser.add_argument_group(
        'stats', 'parameters for report statistics.')

    stats.add_argument(
        '--confidence-level',
        help='confidence level for intervals (aka alpha). (default: 0.95)',
        type=float,
        default=0.95)

    stats.add_argument(
        '--n-kfold-iterations',
        help='number of times to conduct k-fold cross validation. Also \
            changes random state for shuffling n times. (default: 1)',
        type=int,
        default=1)

    figures = parser.add_argument_group('figures', 'parameters for figures.')

    figures.add_argument(
        '--bar-chart-path',
        help='path to save bar chart. (default: ./tex/figures/bar_chart.svg)',
        type=str,
        default='./tex/figures/bar_chart.svg')

    # figures.add_argument(
    #     '--learning-curve-path',
    #     help='path to save learning curve. (default: ./tex/figures/learning_curve.svg)',
    #     type=str,
    #     default='./tex/figures/learning_curve.svg')

    # figures.add_argument(
    #     '--learning-curve-title',
    #     help='title for train-validation plot. (default: Learning Curve for Model)',
    #     type=str,
    #     default='Learning Curve for Model')

    # figures.add_argument(
    #     '--learning-curve-scatter',
    #     choices=[True, False],
    #     help='True for scatter plotted learning curve, False for smooth curve. \
    #         (default: False)',
    #     type=lambda x: bool(strtobool(x)),
    #     default=False)

    figures.add_argument(
        '--bar-plot-title',
        help='title for bar plot. (default: N, K-Fold CV Model Performance Comparison)',
        type=str,
        default='N, K-Fold CV Model Performance Comparison')

    figures.add_argument(
        '--bar-width',
        help='width of bars in bar charts. (default: 0.25)',
        type=float,
        default=0.25)

    return parser


if __name__ == '__main__':
    main()
