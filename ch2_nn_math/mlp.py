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

    def __init__(self, input_dims: int, num_units: int):
        """Define state for neural network layer.

        Args:
            input_dims: Number of units in the previous layer.
            num_units: Number of hidden units.
        """

        # Initialize weight matrix
        self.W = self.glorot_uniform(
            input_dims=input_dims, num_units=num_units)

        # Initialize bias vector
        self.b = np.zeros(shape=(num_units))

        # Initialize list of neurons
        self.hidden_units = [Neuron for j in num_units]

    def glorot_uniform(self, input_dims: int, num_units: int) -> np.ndarray:
        """(Xavier) Glorot uniform initializer.

        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        Args:
            input_dims: Dimensions of previous layer.
            num_units: Number of hidden units (i.e., output dims).

        Returns:
            Array for weight initializer.
        """

        return np.random.uniform(-1/np.sqrt(input_dims), 1/np.sqrt(input_dims),
                                 size=(input_dims, num_units))

    def __call__(self, inputs: np.ndarray):
        """Layer computation.

        Should compute output for each `j` neuron and for all `i`
        in `inputs` vector. 

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

        # Define layers
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
