"""Module for multilayer perceptron from scratch.

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
Simple Backprop: https://www.youtube.com/watch?v=khUVIZ3MON8&t=20s
Derivative of Activation Fxns: https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union

import numpy as np


class Operation(metaclass=ABCMeta):
    """Abstract class for operations with derivatives needed for backprop."""

    @abstractmethod
    def derivative(
            self,
            inputs: Union[tuple[np.ndarray, np.ndarray],
                          np.ndarray]) -> np.ndarray:
        """Call using derivative of operation.

        Args:
            inputs: Inputs for use during operation derivative call.
                Tuple for error functions, otherwise single input.

        Returns:
            Vector of derived outputs.
        """

        pass

    @abstractmethod
    def __call__(
            self,
            inputs: Union[tuple[np.ndarray, np.ndarray],
                          np.ndarray]) -> np.ndarray:
        """Normal call of operation.

        Args:
            inputs: Inputs for use during operation call.
                Tuple for error functions, otherwise single input.

        Returns:
            Vector of outputs.
        """

        pass


class ReLU(Operation):
    """ReLU activation function.

    https://en.wikipedia.org/wiki/Rectifier_(neural_networks) and
    @MISC {333400,
        TITLE = {What is the derivative of the ReLU activation function?},
        AUTHOR = {Jim (https://stats.stackexchange.com/users/67042/jim)},
        HOWPUBLISHED = {Cross Validated},
        NOTE = {URL:https://stats.stackexchange.com/q/333400 (version: 2018-03-15)},
        EPRINT = {https://stats.stackexchange.com/q/333400},
        URL = {https://stats.stackexchange.com/q/333400}
    }
    """

    def derivative(self, inputs: np.ndarray) -> np.ndarray:

        # Compute a boolean tensor where the elements
        # that are greater than 0 are set to 1 while
        # elements <= 0 are set to 0.
        greater_than_zero_tensor = np.greater(inputs, 0).astype(np.float64)
        return greater_than_zero_tensor

    def __call__(self, inputs: np.ndarray) -> np.ndarray:

        # Compute a boolean tensor where the elements
        # that are greater than 0 are set to true while
        # elements <= 0 are set to false.
        greater_than_zero_tensor = np.greater(inputs, 0)
        return greater_than_zero_tensor * inputs


class Sigmoid(Operation):
    """Sigmoid activation function.

    @MISC {1225116,
        TITLE = {Derivative of sigmoid function $\sigma (x) = \frac{1}{1+e^{-x}}$},
        AUTHOR = {Michael Percy (https://math.stackexchange.com/users/229646/michael-percy)},
        HOWPUBLISHED = {Mathematics Stack Exchange},
        NOTE = {URL:https://math.stackexchange.com/q/1225116 (version: 2017-09-01)},
        EPRINT = {https://math.stackexchange.com/q/1225116},
        URL = {https://math.stackexchange.com/q/1225116}
    }
    """

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return self(inputs=inputs) * (1 - self(inputs=inputs))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))


class MeanSquaredError(Operation):
    """Mean squared error cost (loss) function.

    The predictions are the activations of the network. The order of
    arguments in the `derivative` was based on
    `Four fundamental equations behind backpropagation` from
    Nielsen (Ch.2, 2015).
    """

    def derivative(
            self,
            inputs: tuple[np.ndarray, np.ndarray]) -> np.float64:

        targets, predictions = inputs
        return 2 * np.mean(predictions - targets)

    def __call__(
            self,
            inputs: tuple[np.ndarray, np.ndarray]) -> np.float64:

        targets, predictions = inputs
        return np.mean(np.square(targets - predictions))


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
            x: Input matrix.

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
        else:
            activation_a = weighted_input_z

        # Result of layer computation
        return activation_a, weighted_input_z


class MLP:
    """Feedforward neural net with single hidden layer (Multilayer Perceptron)."""

    def __init__(
            self,
            input_dims: int,
            hidden_units: int,
            targets: int,
            learning_rate: float,
            loss_function: str = 'mse',
            l_layers: int = 1,
            hidden_activation: str = 'relu',
            target_activation: Optional[str] = None,):
        """Define state for Multilayer Perceptron.

        The parameters (params) of this hypothesis function are denoted
        in the literature as theta. Therefore, the MLP is a hypothesis
        function parametrized by the weights and biases.

        Args:
            input_dims:
            hidden_units: Number of neurons in hidden layer.
            targets: Target dimensional output.
            loss_function: Specify loss function.
                NOTE: Only supports 'mse'.
            learning_rate: Learning rate(eta) for weight updates.
            hidden_activation: Activation function for hidden layers.
                NOTE: Only supports 'relu'.
            target_activation: Activation function for target layers.
                NOTE: Only support 'sigmoid' or None for now.
        """

        # Save args
        self.learning_rate = learning_rate
        self.l_layers = l_layers

        # Set functions
        if target_activation == 'sigmoid':
            target_activation = Sigmoid()

        if hidden_activation == 'relu':
            hidden_activation = ReLU()
        else:
            raise NotImplementedError

        if loss_function == 'mse':
            self.loss_function = MeanSquaredError()
        else:
            raise NotImplementedError

        # Define layers
        self.hidden = DenseLayer(
            input_dims=input_dims,
            num_units=hidden_units,
            activation_function=hidden_activation)

        self.deep_hidden = [
            DenseLayer(
                input_dims=hidden_units,
                num_units=hidden_units,
                activation_function=hidden_activation)
            for lyr in range(l_layers - 1)]

        self.output = DenseLayer(
            input_dims=hidden_units,
            num_units=targets,
            activation_function=target_activation)

        self.sequential = [self.hidden, *self.deep_hidden, self.output]

        # The i^th element of each of these caches corresponds
        # to the outputs of the l^th layer...
        # for two layers (hidden and output) there are only
        # two elements of each cache.
        self.activations_cache = []
        self.weighted_inputs_cache = []

    @property
    def cache(self,) -> tuple[list[float], list[float]]:
        """Returns activation and weighted inputs caches."""

        return self.activations_cache, self.weighted_inputs_cache

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

        # Dump cache
        self._clear_cache()

        # Call layers in model and cache layer outputs
        activations = inputs
        for lyr in self.sequential:
            activations, weighted_inputs = lyr(activations)
            self._cache(activations=activations,
                        weighted_inputs=weighted_inputs)

        # Result of forward pass
        return activations

    def _cache(self, activations: np.ndarray, weighted_inputs: np.ndarray) -> None:
        """Caches activations and weighted inputs from layer for backprop.

        Args:
            activations:
            weighted_inputs:
        """

        self.activations_cache.append(activations)
        self.weighted_inputs_cache.append(activations)

    def _clear_cache(self,) -> None:
        """Sets cache lists to empty."""

        self.activations_cache = []
        self.weighted_inputs_cache = []

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

    def _compute_output_lyr_error(self, ):
        """Computes delta for output layer for backprop."""

        return

    def _backpropagation(self,) -> np.ndarray:
        """Compute the gradient."""
        pass

    def _gradient_descent(self,) -> None:
        """Uses gradient to minimize loss."""
        pass
