"""Module for multilayer perceptron (single hidden layer) from scratch.

General Backprop (Goodfellow et al., Deep Learning 6.5.6 p. 211):
    Should each element of the weight matrix and bias vector
    be a <class `Parameter`> in order to implement the operations
    described for general backprop on the above page?

Deep Learning with Python 2ed (pp. 26-67, 2021)
Goodfellow et al. Deep Learning  (Ch. 6.5 pp. 200-220, 2016)
Wiki: https://en.wikipedia.org/wiki/Backpropagation
ML Mastery Backprop: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
Climkovic & Jl: Neural Networks & Back Propagation Algorithm (Academia, 2015)
Nielsen (Ch. 2, 2015) http://neuralnetworksanddeeplearning.com/chap2.html
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

import numpy as np


class Activation(metaclass=ABCMeta):
    """Abstract class for activation functions."""

    @abstractmethod
    def derivative_call(self, inputs: np.ndarray) -> np.ndarray:
        """Call using derivative of activation function.

        Args:
            inputs: Vector of inputs.

        Returns:
            Vector of derived outputs.
        """

        pass

    @abstractmethod
    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Normal call for activation function.

        Args:
            inputs: Vector of inputs.

        Returns:
            Vector of activated outputs.
        """

        pass


class Sigmoid(Activation):
    """Sigmoid activation function."""

    def derivative_call(self, inputs: np.ndarray) -> np.ndarray:
        """
        @MISC {1225116,
            TITLE = {Derivative of sigmoid function $\sigma (x) = \frac{1}{1+e^{-x}}$},
            AUTHOR = {Michael Percy (https://math.stackexchange.com/users/229646/michael-percy)},
            HOWPUBLISHED = {Mathematics Stack Exchange},
            NOTE = {URL:https://math.stackexchange.com/q/1225116 (version: 2017-09-01)},
            EPRINT = {https://math.stackexchange.com/q/1225116},
            URL = {https://math.stackexchange.com/q/1225116}
        }
        """
        return self.call(inputs=inputs) * (1 - self.call(inputs=inputs))

    def call(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))


class DenseLayer:
    """A densely connected layer in a neural network."""

    def __init__(
            self,
            input_dims: int,
            num_units: int,
            activation_function: Optional[Callable] = None):
        """Define state for neural network layer.

        Args:
            input_dims: Number of units in the previous layer.
            num_units: Number of hidden units.
            activation_function: Activation function for neurons.
        """

        # Save function arg
        self.activation_function = activation_function

        # Initialize weight matrix
        self.W = self.glorot_uniform(
            input_dims=input_dims, num_units=num_units)

        # Initialize bias vector
        self.b = np.zeros(shape=(num_units))

    def glorot_uniform(self, input_dims: int, num_units: int) -> np.ndarray:
        """(Xavier) Glorot uniform initializer.

        http: // proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        Args:
            input_dims: Dimensions of previous layer.
            num_units: Number of hidden units(i.e., output dims).

        Returns:
            Array for weight initializer.
        """

        return np.random.uniform(-1/np.sqrt(input_dims), 1/np.sqrt(input_dims),
                                 size=(input_dims, num_units))

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute layer activations and weighted inputs.

        Args:
            x: Input vector(1 BY num_ele_in_x).

        Returns:
            Activation vector and weighted inputs vector.
        """

        # Transpose to row vector for single sample
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        # Affine transformation
        weighted_input_z = np.dot(x, self.W) + self.b

        # Optional activation function
        if self.activation_function is not None:
            activation_a = np.apply_along_axis(
                self.activation_function, axis=-1, arr=weighted_input_z)

        # Result of layer computation
        return activation_a, weighted_input_z


class MLP:
    """Feedforward neural net with single hidden layer (Multilayer Perceptron)."""

    def __init__(
            self,
            input_dims: int,
            hidden_units: int,
            targets: int,
            loss_function: Callable,
            learning_rate: float,
            hidden_activation: Optional[Callable] = None,
            target_activation: Optional[Callable] = None,):
        """Define state for Multilayer Perceptron.

        The parameters (params) of this hypothesis function are denoted
        in the literature as theta. Therefore, the MLP is a hypothesis
        function parametrized by the weights and biases.

        Args:
            input_dims:
            hidden_units: Number of neurons in hidden layer.
            targets: Target dimensional output.
            loss_function: Function object for loss computations.
            learning_rate: Learning rate(eta) for weight updates.
            hidden_activation: Activation function for hidden layers.
            target_activation: Activation function for target layers.
        """

        # Save args
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        # Define layers
        self.hidden = DenseLayer(
            input_dims=input_dims,
            num_units=hidden_units,
            activation_function=hidden_activation)

        self.output = DenseLayer(
            input_dims=hidden_units,
            num_units=targets,
            activation_function=target_activation)

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int) -> None:
        """Fit the MLP to data.

        Args:
            x: Input data with `n` samples and `m` features.
            y: Target data with `n` samples, and `m` features.
            batch_size: Size of batch for mini-batch gradient descent.
                Drops remainder batch by default.
            epochs: Number of epochs to train neural network.
        """

        # Get the number of samples
        samples = x.shape[0]

        # Get batch indices
        batch_indices = np.random.choice(
            a=samples, size=(samples//batch_size, batch_size), replace=False)

        # Batch the data
        batch_data = zip(x[batch_indices], y[batch_indices])

        # Training loop
        for epoch in epochs:
            for batch_step, (x_batch, y_batch) in enumerate(batch_data):
                preds = self._forward_pass(x_batch)
                loss = self._compute_loss(y_true=y_batch, y_pred=preds)
                grads = self._backpropagation()
                self._gradient_descent()

    def _forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward pass through network."""

        # Call hidden layer
        hidden_output = self.hidden(inputs)
        targets = self.output(hidden_output)

        # Result of forward pass
        return targets

    def _backward_pass(self, loss: np.float64, ):
        """"""
        pass

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """Compute loss.

        Args:
            y_true: Targets.
            y_pred: Predictions.

        Returns:
            Scalar loss.
        """

        return self.loss_function(y_true, y_pred)

    def _backpropagation(self,) -> np.ndarray:
        """Compute the gradient."""
        pass

    def _gradient_descent(self,) -> None:
        """Uses gradient to minimize loss."""
        pass
