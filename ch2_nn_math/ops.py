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
                Tuple (target, pred) for error functions,
                otherwise single input.

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
        return 1 / (1 + np.exp(-1 * inputs))


class Linear(Operation):
    """Linear activation function."""

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.ones_like(inputs)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs


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
        return super().derivative(inputs)

    def gradient(
            self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Computes the gradient with respect to activation (preds).

        This is a vectorized function and is called on each element of 
        an activation vector in order to compute the partial derivative
        of the cost with respect to the j^{th} activation for the 
        l^{th} layer.

        Args:
            inputs: Targets, predictions vectors.

        Returns:
            Vector (gradient) of values.
        """

        targets, predictions = inputs
        return 2 * (predictions - targets)

    def __call__(
            self,
            inputs: tuple[np.ndarray, np.ndarray]) -> np.float64:
        """Compute cost given inputs.
        
        Args:
            inputs: Targets and predictions vectors.

        Return:
            Scalar cost.
        """

        targets, predictions = inputs
        return np.mean(np.square(targets - predictions))


class BinaryCrossEntropy(Operation):
    """Binary cross entropy loss (cost) function."""

    def __init__(self,):
        """Initializes sigmoid function for binary cross entropy."""

        self.sigmoid = Sigmoid()

    def derivative(self, inputs: Union[tuple[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        return super().derivative(inputs)

    def gradient(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """"""
        pass

    def __call__(self, 
        inputs: tuple[np.ndarray, np.ndarray],
        from_logits: bool = False) -> np.ndarray:
        """Compute cost given inputs.
        
        Args:
            inputs: Targets and predictions vectors. 
                Assumes predictions are not from logits.
            from_logits: True for logits, false for normalized log 
                probabilities (i.e., used sigmoid activation function).
                Assumes not from logits.

        Return:
            Scalar cost.
        """
        
        targets, predictions = inputs

        if from_logits:
            predictions = self.sigmoid(predictions)

        return -1 * np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    


class CategoricalCrossEntropy(Operation):
    """Categorical cross entropy (aka, NLL) loss (cost) function."""

    def derivative(self, inputs: Union[tuple[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        return super().derivative(inputs)

    def __call__(self, inputs: Union[tuple[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        return super().__call__(inputs)
