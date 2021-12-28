"""Module for multilayer perceptron (single hidden layer) from scratch.

Could make layer shapes dynamic.

General Backprop (Goodfellow et al., Deep Learning 6.5.6 p. 211):
    Should each element of the weight matrix and bias vector
    be a <class `Parameter`> in order to implement the operations
    described for general backprop on the above page?

Deep Learning with Python 2ed (pp. 26-67)
Goodfellow et al. Deep Learning  (Ch. 6.5 pp. 200-220)
https://en.wikipedia.org/wiki/Backpropagation
"""

from typing import Callable, Optional

import numpy as np


class AffineTransform:
    """Operation class for affine transformation (XW + b)."""

    def __init__(self,):
        """"""
        pass


class Parameter:
    """Variable (parameter) to be updated by backprop."""

    def __init__(self,):
        """"""
        pass

    def get_operation(self,):
        """"""
        pass

    def get_consumers(self,):
        """"""
        pass

    def get_inputs(self,):
        """"""
        pass


class DenseLayer:
    """A densely connected layer in a neural network."""

    def __init__(
            self,
            input_dims: int,
            num_units: int,
            activation_function: Callable = None):
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

        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

        Args:
            input_dims: Dimensions of previous layer.
            num_units: Number of hidden units (i.e., output dims).

        Returns:
            Array for weight initializer.
        """

        return np.random.uniform(-1/np.sqrt(input_dims), 1/np.sqrt(input_dims),
                                 size=(input_dims, num_units))

    def activate(
            self,
            x: np.ndarray,
            W: np.ndarray,
            b: np.ndarray,
            activation_function: Callable = None) -> np.float64:
        """Activation of hidden units.

        Args:
            x: Input vector (1 BY num_ele_in_x).
            W: Weight matrix (num_ele_in_x BY num_hidden_units).
            b: Bias vector (num_hidden_units).

        Returns:
            Scalar output of activation.
        """

        # Transpose to row vector for single sample
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        # Affine transformation
        transform = np.dot(x, W) + b

        # Optional activation function
        if activation_function is not None:
            transform = np.apply_along_axis(
                activation_function, axis=-1, arr=transform)

        # Result of activation
        return transform

    def __call__(self, inputs: np.ndarray):
        """Layer computation.

        Should compute output for each `j` neuron and for all `i`
        in `inputs` vector.

        Args:
            inputs: Vector of inputs from previous layer.

        Returns:
            Vector of outputs from neurons.
        """

        # The layer call should activate the hidden units abstraction
        layer_output = self.activate(
            x=inputs,
            W=self.W,
            b=self.b,
            activation_function=self.activation_function)

        # Result of layer call
        return layer_output


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
        function parametrized by the weights (edges) of the weight matrix
        and the bias vector.

        Args:
            input_dims:
            hidden_units: Number of neurons in hidden layer.
            targets: Target dimensional output.
            loss_function: Function object for loss computations.
            learning_rate: Learning rate (eta) for weight updates.
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

    def _backward_pass(self,):
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
