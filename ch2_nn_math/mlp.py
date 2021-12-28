"""Module for multilayer perceptron (single hidden layer) from scratch.

Deep Learning with Python 2ed (pp. 26-67)
Goodfellow et al. Deep Learning  (Ch. 6.5 pp. 200-220)
"""

import numpy as np


class Neuron:
    """A single hidden unit."""

    def __init__(self, activation_function: function = None):
        """Define state for neural network neuron.

        Args:
            activation_function: Activation function for neuron call.
                Defaults to None (i.e., `XW + b` activation)
        """

        self.activation_function = activation_function

    def activate(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.float64:
        """Activation of hidden unit.

        Args:
            x: Input vector (1 BY num_ele_in_x).
            W: Weight matrix (num_ele_in_x BY num_hidden_units).
            b: Bias vector (num_hidden_units). 

        Returns:
            Scalar output of activation.
        """

        # x^T @ W + b
        geometric_transformation = np.dot(np.expand_dims(x, axis=0), W) + b

        # Optional activation function
        if self.activation_function is not None:
            geometric_transformation = self.activation_function(
                geometric_transformation)

        # Result of activation
        return geometric_transformation


class Layer:
    """A layer in the neural network."""

    def __init__(self, num_units: int):
        """Define state for neural network layer.

        Args:
            num_units: Number of hidden units.
        """
        pass

    def __call__(self, inputs: np.ndarray):
        """Layer computation.

        Args:
            Vector of inputs from previous layer.

        Returns:
            Vector of outputs from neurons.
        """
        pass


class MLP:
    """Feedforward neural net with single hidden layer (Multilayer Perceptron)."""

    def __init__(
            self,
            hidden_units: int,
            targets: int,
            loss_function: function,
            learning_rate: float):
        """Define state for Multilayer Perceptron.

        The parameters (params) of this hypothesis function are denoted
        in the literature as theta. Therefore, the MLP is a hypothesis
        function parametrized by the weights (edges) of the weight matrix
        and the bias vector.

        Args:
            hidden_units: Number of neurons in hidden layer.
            targets: Target dimensional output.
            loss_function: Function object for loss computations.
            learning_rate: Learning rate (eta) for weight updates.
        """

        # Initialize the weight matrix
        pass

        # Initialize the bias vector
        pass

    def forward_pass(self,) -> np.ndarray:
        """Perform forward pass through network."""
        pass

    def compute_loss(self,) -> np.float64:
        """Compute loss."""
        pass

    def backpropagation(self,):
        """Compute the gradient."""
        pass

    def stochastic_gradient_descent(self,):
        """Uses gradient to minimize loss."""
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, ) -> None:
        """Fit the MLP to data.

        TODO: Flatten x and y.
        """
        pass
