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

    def activate(self, x: np.array, W: np.array, b: np.array) -> np.float64:
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

        # Optional activation
        if self.activation_function is not None:
            geometric_transformation = self.activation_function(
                geometric_transformation)

        return geometric_transformation


class Layer:
    """"""

    def __init__(self, num_units: int):
        """"""

    def __call__(self, inputs):
        """"""
        pass


class MLP:
    """Feedforward neural net with single hidden layer (Multilayer Perceptron)."""

    def __init__(self, hidden_units: int, targets: int):
        """Define state for Multilayer Perceptron.

        The parameters (param) of this hypothesis function are denoted
        in the literature as theta. Therefore, the MLP is a hypothesis
        function parametrized by the weights (edges) of the weight matrix
        and the bias vector. For vector input, the weight "matrix" is not
        a matrix but rather a vector only.

        Args:
            hidden_units: Number of neurons in hidden layer.
            targets: Target dimensional output.
        """

        # Initialize the weight matrix
        pass

        # Initialize the bias vector
        pass

    def forward_pass(self,):
        """"""
        pass

    def compute_loss(self,):
        """"""
        pass

    def compute_gradient(self,):
        """"""
        pass

    def backpropagate(self,):
        """"""
        pass
