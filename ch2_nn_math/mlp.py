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

from __future__ import annotations
from typing import Callable, Optional

import numpy as np

from ops import Operation, Sigmoid, ReLU, Linear, MeanSquaredError


class DenseLayer:
    """A densely connected layer in a neural network."""

    def __init__(
            self,
            input_dims: int,
            num_units: int,
            activation_function: Optional[Callable] = Linear):
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
        activation_a = np.apply_along_axis(
            self.activation_function, axis=-1, arr=weighted_input_z)

        # Result of layer computation
        return activation_a, weighted_input_z


class MLP:
    """Feedforward neural net (Multilayer Perceptron)."""

    def __init__(
            self,
            input_dims: int,
            hidden_units: int,
            targets: int,
            learning_rate: float,
            l_layers: int = 1,
            loss_function: str = 'mse',
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
            learning_rate: Learning rate(eta) for weight updates.
            l_layers: Number of layers
            loss_function: Specify loss function.
                NOTE: Only supports 'mse'.
            hidden_activation: Activation function for hidden layers.
                NOTE: Only supports 'relu'.
            target_activation: Activation function for target layers.
                NOTE: Only support 'sigmoid' or None for now.
        """

        # Save args
        self.learning_rate = learning_rate
        self.l_layers = l_layers

        # Set final layer activation function
        if target_activation is None:
            self.target_activation = Linear()
        elif target_activation == 'sigmoid':
            self.target_activation = Sigmoid()
        else:
            raise ValueError

        # Set hidden layer activation functions
        if hidden_activation is None:
            self.hidden_activation = Linear()
        elif hidden_activation == 'relu':
            self.hidden_activation = ReLU()
        else:
            raise ValueError

        # Set loss function
        if loss_function == 'mse':
            self.loss_function = MeanSquaredError()
        else:
            raise ValueError

        # Define layers
        self.hidden = DenseLayer(
            input_dims=input_dims,
            num_units=hidden_units,
            activation_function=self.hidden_activation)

        self.deep_hidden = [
            DenseLayer(
                input_dims=hidden_units,
                num_units=hidden_units,
                activation_function=self.hidden_activation)
            for lyr in range(l_layers - 1)]

        self.output = DenseLayer(
            input_dims=hidden_units,
            num_units=targets,
            activation_function=self.target_activation)

        # Sequential model with a None for placeholder for input layer
        self.sequential = [None, self.hidden, *self.deep_hidden, self.output]

        # The i^th element of each of these caches corresponds
        # to the outputs of the l^th layer...
        # for two layers (hidden and output) there are only
        # two elements of each cache.
        self.activations_cache = []
        self.weighted_inputs_cache = []

    @property
    def cache(self,) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Returns activation and weighted inputs caches."""

        return self.activations_cache, self.weighted_inputs_cache

    @property
    def num_layers(self,) -> int:
        """Returns the number of layers (includes the input layer)."""

        return len(self.sequential)

    @property
    def layers(self,) -> list[DenseLayer]:
        """Returns a list of dense layers in the network."""

        return self.sequential[1:]

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
        for epoch in range(epochs):
            for batch_step, (x_batch, y_batch) in enumerate(batch_data):

                print(f'{batch_step}/{samples//batch_size}')
                print(x_batch.shape, x_batch)
                print(y_batch.shape, y_batch)

                preds = self._forward_pass(x_batch)

                print('Preds:')
                print(preds.shape)
                print(preds)

                loss = self._compute_loss(y_true=y_batch, y_pred=preds)

                print('Loss:')
                print(loss)

                weight_grads, bias_grads = self._backward_pass(
                    y_true=y_batch)

                print('Weight Grads:')
                print(len(weight_grads))
                print(weight_grads)
                print('Bias Grads:')
                print(len(bias_grads))

                breakpoint()

                self._gradient_descent(
                    weight_grads=weight_grads, bias_grads=bias_grads)

    def _forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward pass through network."""

        # Dump cache
        self._clear_cache()

        # Call layers in model and cache layer output
        # The inputs have no weight matrix associated with them...
        # but the inputs themselves are treated as activations for backprop
        # purposes.
        activations = inputs
        self._cache(activations=activations, weighted_inputs=None)
        for lyr in range(1, self.num_layers):
            activations, weighted_inputs = self.sequential[lyr](activations)
            self._cache(activations=activations,
                        weighted_inputs=weighted_inputs)

        # Result of forward pass
        return activations

    def _compute_loss(
            self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """Compute scalar loss.

        Args:
            y_true: Vectors of targets.
            y_pred: Vector of predictions.

        Returns:
            Scalar loss.
        """

        return self.loss_function((y_true, y_pred))

    def _backward_pass(
            self,
            y_true: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute the gradient of the cost function.

        After forward the pass occurs, use the cached outputs
        to compute cost function gradients. The algorithm proceeds
        as follows:

        1. delta^{L} = grad_a(C) * deriv_activation(weighted_input_of_lyr)
                     = `self._compute_output_layer_error`.
                     = dCost/dBias
            1.1 Compute cost w.r.t weight matrix using delta^{L}.
        2. for L - 1 to 0:  # This is 'backpropagation' part.
            2.1 Compute delta^{l} \
                == dCost/dBias for the current layer
                == self._compute_hidden_layer_error
            2.2  Compute cost w.r.t weight matrix using delta^{l}
        3. Make sure to save the delta computations for biases
            and weights for each layer as these will be needed
            to update the weights.

        Args:
            y_true: The ground truth needed for the final layer L.

        Returns:
            Tuple of lists of weight errors and bias errors for each layer.
            The length of each list is equal to the number of layers
            (including the input layer). Therefore, the first element of the
            lists will be None since there are no errors for the input layer.
        """

        # Get cached activations and weighted inputs
        activations, weighted_inputs = self.cache

        # Compute errors for bias and then weight matrices
        delta_L = self._compute_output_layer_error(
            output_activations=activations[-1],
            y_true=y_true,
            wted_input_of_final_lyr=weighted_inputs[-1])

        dCost_dW_L = self._compute_deriv_cost_wrt_wt(
            activations_prev_lyr=activations[-2],
            delta_cur_lyr=delta_L)

        # Save deltas
        deltas = [None for i in range(self.num_layers)]
        deltas[-1] = delta_L

        dCost_dWs = [None for i in range(self.num_layers)]
        dCost_dWs[-1] = dCost_dW_L

        # Backpropagate error through layers...
        # Must use `self.num_layers-2` because `len(lst)-1` is index `L`
        # and iteration begins at `L-2`
        # TODO: Issue with indexing
        for lyr in range(self.num_layers-2, 0, -1):

            print('Layer:', lyr)

            # Compute errors
            delta_l = self._compute_hidden_layer_error(
                wt_matrix_of_lyr_plus_one=self.sequential[lyr+1].W,
                delta_of_lyr_plus_one=deltas[lyr+1],
                wted_input_of_cur_lyr=weighted_inputs[lyr],
                hidden_activation=self.sequential[lyr].activation_function)

            dCost_dW_l = self._compute_deriv_cost_wrt_wt(
                activations_prev_lyr=activations[lyr-1],
                delta_cur_lyr=delta_l)

            # Update lists
            dCost_dWs[lyr] = dCost_dW_l
            deltas[lyr] = delta_l

        # Return the error lists
        return dCost_dWs, deltas

    def _gradient_descent(
            self,
            weight_grads: list[np.ndarray],
            bias_grads: list[np.ndarray]) -> None:
        """Uses gradient to minimize loss.

        Args:
            weight_grads: List of weight gradient vectors
            bias_grads: List of bias gradient vectors.
        """

        # Zipped model and gradients iterator
        grad_iterator = zip(self.sequential, weight_grads, bias_grads)

        # Skip the first element (None entry for input layer)
        next(grad_iterator)

        # Update gradients
        for lyr, wt_grad, bias_grad in grad_iterator:

            print(type(lyr))
            print(type(wt_grad), wt_grad)
            print(type(bias_grad), bias_grad)
            breakpoint()

            lyr.W -= self.learning_rate * np.mean(wt_grad, axis=0)
            lyr.b -= self.learning_rate * np.mean(bias_grad, axis=0)

    def _cache(self, activations: np.ndarray, weighted_inputs: np.ndarray) -> None:
        """Caches activations and weighted inputs from layer for backprop.

        Args:
            activations:
            weighted_inputs:
        """

        self.activations_cache.append(activations)
        self.weighted_inputs_cache.append(weighted_inputs)

    def _clear_cache(self,) -> None:
        """Sets cache lists to empty."""

        self.activations_cache = []
        self.weighted_inputs_cache = []

    def _compute_output_layer_error(
            self,
            output_activations: np.ndarray,
            y_true: np.ndarray,
            wted_input_of_final_lyr: np.ndarray) -> np.ndarray:
        """Computes delta for output layer for backprop.

        Args:
            output_activations: Predictions (activations `a`) of the final layer.
            y_true: Ground truth.
            wted_input_of_final_lyr: Weighted input for the final layer in network.

        Returns:
            Delta vector of the output layer.
        """

        grad_cost_wrt_activation = self.loss_function.gradient(
            (output_activations, y_true))
        deriv_of_activation_of_wted_input_of_final_lyr = self.target_activation.derivative(
            wted_input_of_final_lyr)
        delta_lyr = grad_cost_wrt_activation * \
            deriv_of_activation_of_wted_input_of_final_lyr

        return delta_lyr

    def _compute_hidden_layer_error(
            self,
            wt_matrix_of_lyr_plus_one: np.ndarray,
            delta_of_lyr_plus_one: np.ndarray,
            wted_input_of_cur_lyr: np.ndarray,
            hidden_activation: Operation) -> np.ndarray:
        """Uses the `l+1` and `l` layer in cache to compute delta vector.

        NOTE: This is equivalent to the rate of change of the loss (cost)
        function with respect to the layer's bias vector.

        Args:
            wt_matrix_of_lyr_plus_one: Weight matrix of next layer.
            delta_of_lyr_plus_one: Delta (error) of next layer.
            wted_input_of_cur_lyr: Weighted input `z` of current layer.
            hidden_activation: Activation function of hidden layer.

        Returns:
            The delta vector of the current layer.
        """

        wted_err = np.dot(wt_matrix_of_lyr_plus_one, delta_of_lyr_plus_one)
        wted_derived_activation_err = wted_err * \
            hidden_activation.derivative(wted_input_of_cur_lyr)

        return wted_derived_activation_err

    def _compute_deriv_cost_wrt_wt(
            self,
            activations_prev_lyr: np.ndarray,
            delta_cur_lyr: np.ndarray) -> np.ndarray:
        """Computes derivative of cost fxn with respect to layer weight.

        Uses activation of previous layer and delta of current layer to
        compute dC/dw_jk.

        TODO: Is dot product appropriate here?

        Args:
            activations_prev_lyr: Activations of previous layer.
            delta_cur_lyr:

        Returns:
            Derivative cost w.r.t to weight matrix vector.
        """

        return np.dot(activations_prev_lyr, delta_cur_lyr)
