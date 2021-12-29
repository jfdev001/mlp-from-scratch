"""Module for handling neural network operations and their derivatives."""


from abc import ABCMeta, abstractmethod
from typing import Union

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
    """(Auto-vectorized) sigmoid activation function.

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
    Nielsen (Ch.2, 2015). Similarly, the gradient calculation in BP1a of 
    is described in the same resource.
    """

    def derivative(
            self,
            inputs: tuple[np.ndarray, np.ndarray]) -> np.float64:

        targets, predictions = inputs
        return 2 * np.mean(predictions - targets)

    def gradient(
            self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Computes the gradient vector with respect to activations (preds)."""

        targets, predictions = inputs
        return 2 * (predictions - targets)

    def __call__(
            self,
            inputs: tuple[np.ndarray, np.ndarray]) -> np.float64:

        targets, predictions = inputs
        return np.mean(np.square(targets - predictions))
