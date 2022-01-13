"""Module for multilayer perceptron from scratch.

General Backprop (Goodfellow et al., Deep Learning 6.5.6 p. 211):
    Should each element of the weight matrix and bias vector
    be a <class `Parameter`> in order to implement the operations
    described for general backprop on the above page?

NOTE: The weight matrix is of shape (output_dims, input_dims)

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
from collections import defaultdict
from typing import Callable, Optional, List, Tuple, DefaultDict

import numpy as np
from numpy.random.mtrand import logseries

from ops import Operation, Sigmoid, ReLU, Linear, MeanSquaredError


class DenseLayer:
    """A densely connected layer in a neural network."""

    def __init__(
            self,
            input_dims: int,
            num_units: int,
            activation_function: Optional[Callable] = Linear,
            debug: Optional[bool] = False):
        """Define state for neural network layer.

        Args:
            input_dims: Number of units in the previous layer.
            num_units: Number of hidden units.
            activation_function: Activation function for neurons.
            debug: Bool for printing debug information.
        """

        # Save function arg
        self.activation_function = activation_function

        self.debug = debug

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
                                 size=(num_units, input_dims))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute layer activations and weighted inputs.

        Args:
            x: Input matrix.

        Returns:
            Activation vector and weighted inputs vector.
        """

        # Make row vector for single sample
        x = np.atleast_2d(x)

        # shapes
        if self.debug:
            print('__call__ inputs')
            print('x:', x.shape, '@ W^{T}:',
                  self.W.shape[::-1], '+ b', self.b.shape)
            breakpoint()

        # Affine transformation
        # z_{m x n_h}
        weighted_input_z = np.dot(x, np.transpose(self.W)) + self.b

        # Activation function
        # a_{m x n_h}
        activation_a = np.apply_along_axis(
            self.activation_function, axis=-1, arr=weighted_input_z)

        # Shapes
        if self.debug:
            print('__call__ transformations')
            print('z:', weighted_input_z.shape)
            print('a:', activation_a.shape)
            breakpoint()

        # Result of layer computation
        # a (samples, hidden units), (samples, hidden units)
        return activation_a, weighted_input_z


class MLP:
    """Feedforward neural net (Multilayer Perceptron)."""

    def __init__(
            self,
            input_dims: int,
            hidden_units: int,
            targets: int,
            learning_rate: float,
            loss_function: Operation[object],
            l_layers: int = 1,
            hidden_activation: Operation[object] = None,
            target_activation: Operation[object] = None,
            debug: Optional[bool] = False):
        """Define state for Multilayer Perceptron.

        The parameters (params) of this hypothesis function are denoted
        in the literature as theta. Therefore, the MLP is a hypothesis
        function parametrized by the weights and biases.

        Args:
            input_dims: Number of features in training data.
            hidden_units: Number of neurons in hidden layer.
            targets: Number of neurons in output layer.
            learning_rate: Learning rate (eta) for weight updates.
            loss_function: Specify loss function.
            l_layers: Number of hidden layers.
            hidden_activation: Activation function for hidden layers.
            target_activation: Activation function for target layers.
            debug: Bool to debug or not... print shapes of inputs, etc.
        """

        # Save args
        self.learning_rate = learning_rate
        self.l_layers = l_layers
        self.batch_size = None
        self.debug = debug

        # Set default final layer activation function
        if target_activation is None:
            self.target_activation = Linear()
        else:
            self.target_activation = target_activation

        # Set default hidden layer activation functions
        if hidden_activation is None:
            self.hidden_activation = ReLU()
        else:
            self.hidden_activation = hidden_activation

        # Set loss
        self.loss_function = loss_function

        # Validate binary cross entropy
        if self.loss_function.__class__.__name__ == 'BinaryCrossEntropy' \
                and targets != 2:
            raise ValueError(
                'Binary classification problems can have only have 2 target.')

        elif self.loss_function.__class__.__name__ == 'BinaryCrossEntropy' \
                and targets == 2:

            # Output will be probability of sample x belonging to class 1.
            targets = 1

        # Define layers
        self.hidden = DenseLayer(
            input_dims=input_dims,
            num_units=hidden_units,
            activation_function=self.hidden_activation,
            debug=debug)

        self.deep_hidden = [
            DenseLayer(
                input_dims=hidden_units,
                num_units=hidden_units,
                activation_function=self.hidden_activation,
                debug=debug)
            for lyr in range(l_layers - 1)]

        self.output = DenseLayer(
            input_dims=hidden_units,
            num_units=targets,
            activation_function=self.target_activation,
            debug=debug)

        # Sequential model with a None for placeholder for input layer
        self.sequential = [None, self.hidden, *self.deep_hidden, self.output]

        # The i^th element of each of these caches corresponds
        # to the outputs of the l^th layer...
        # for two layers (hidden and output) there are only
        # two elements of each cache.
        self.activations_cache = []
        self.weighted_inputs_cache = []

        # Dictionary {'train_loss': [], 'val_loss': []}
        # where there is a single value per epoch
        self.history = defaultdict(list)

    @property
    def cache(self,) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns activation and weighted inputs caches."""

        return self.activations_cache, self.weighted_inputs_cache

    @property
    def num_layers(self,) -> int:
        """Returns the number of layers (includes the input layer)."""

        return len(self.sequential)

    @property
    def layers(self,) -> List[DenseLayer]:
        """Returns a list of dense layers in the network."""

        return self.sequential[1:]

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            epochs: int) -> DefaultDict[List]:
        """Fit the MLP to data.

        Args:
            x: Input data with `n` samples and `m` features.
            y: Target data with `n` samples, and `m` features.
            batch_size: Size of batch for mini-batch gradient descent.
                Drops remainder batch by default.
            epochs: Number of epochs to train neural network.

        Returns:
            History dictionary with loss values per epoch.
        """

        # Get the number of samples
        num_samples = x.shape[0]

        # Save the batch size for computations later
        self.batch_size = batch_size
        num_batches = num_samples//batch_size

        # Get batch indices
        batch_indices = np.random.choice(
            a=num_samples, size=(num_batches, batch_size), replace=False)

        # Create batch data generator
        # https://stackoverflow.com/questions/50465966/re-using-zip-iterator-in-python-3/50466346
        def batch_data(): return zip(x[batch_indices], y[batch_indices])

        # Training loop
        for epoch in range(epochs):

            # Track losses per epoch over batches
            losses = []
            for batch_step, (x_batch, y_batch) in enumerate(batch_data()):

                # This is a single training step and could be
                # refactored as training iteration

                if self.debug:
                    print('MLP.fit batch step')
                    print(f'{batch_step+1}/{num_batches}')

                # Predictions for sample
                preds = self._forward_pass(x_batch)

                # Loss metric, not used for grad descent
                loss = self._compute_loss(y_true=y_batch, y_pred=preds)
                losses.append(loss)

                # Compute gradients
                weight_grads, bias_grads = self._backward_pass(y_true=y_batch)

                # Use gradient weights to descend cost function
                # (i.e., apply grads)
                self._gradient_descent(
                    weight_grads=weight_grads,
                    bias_grads=bias_grads)

            # Update performance over epoch
            mean_of_batch_losses = np.mean(losses)
            self.history['train_loss'].append(mean_of_batch_losses)
            print(f'Epoch {epoch+1}/{epochs} -- Loss: {mean_of_batch_losses}')

        # Validation loop where predictions and losses only are calculated
        # no gradient descent... this would have to be called per epoch...
        pass

        # Model performances
        return self.history

    def _forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward pass through network.

        Args:
            inputs: Array of x data.

        Returns:
            Predictions (aka activations) given inputs.
        """

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

        if self.debug:
            print('end _forward_pass')
            breakpoint()

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
            y_true: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute the gradient of the cost function.

        After forward the pass occurs, use the cached outputs
        to compute cost function gradients. The algorithm proceeds
        as follows:

        1. delta^{L} = dCost/dBias
            1.1 Compute cost w.r.t weight matrix using delta^{L}.
        2. for L - 1 to 0:  # This is 'backpropagation' part.
            2.1 Compute delta^{l} \
                == dCost/dBias for the current layer
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

        # Make ground truth a row vector if single regression sample...
        # otherwise make into column vector
        if 'Entropy' not in self.loss_function.__class__.__name__:
            y_true = np.atleast_2d(y_true)
            if self.debug:
                print('MLP _backward_pass')
                print(f'Regression Problem...{y_true.shape}')
                print(y_true)
                breakpoint()
        else:
            y_true = np.expand_dims(y_true, axis=-1)
            if self.debug:
                print('MLP _backward_pass')
                print(f'Classification Problem...{y_true.shape}')
                print(y_true)
                breakpoint()

        # One delta_L vector with a number of columns
        # equal to the number of targets for each row (training example)
        # (batch_size, num_targets)
        dCost_dBias_L_samples = np.array([
            self._compute_delta_last_lyr(
                output_activations=activations[-1][sample],
                y_true=y_true[sample, :],
                wted_input_of_final_lyr=weighted_inputs[-1][sample])
            for sample in range(self.batch_size)])

        # One weight cost matrix for each training example
        # (batch_size, num_neurons_in_cur_lyr, num_neurons_in_pre_lyr)
        dCost_dW_L_samples = np.array([
            self._compute_deriv_cost_wrt_wt(
                activations_prev_lyr=activations[-2][sample],
                delta_cur_lyr=dCost_dBias_L_samples[sample])
            for sample in range(self.batch_size)])

        # Save deltas, which are equivalent to partial derivatives
        # with respect to biases for a layer...
        # this will end up being a ragged rank-3 tensor
        # (num_layers, batch_size, num_hidden_units_in_cur_lyr)
        dCost_dBias_lyrs = [None for i in range(self.num_layers)]
        dCost_dBias_lyrs[-1] = dCost_dBias_L_samples

        # Save the partial derivatives with respect to the weights,
        # this will end up being a ragged rank-4 tensor
        # (num_layers, batch_size, num_neurons_in_cur_lyr, num_neurons_in_pre_lyr)
        dCost_dW_lyrs = [None for i in range(self.num_layers)]
        dCost_dW_lyrs[-1] = dCost_dW_L_samples

        # Backpropagate error through layers...
        # Must use `self.num_layers-2` because `len(lst)-1` is index `L`
        # and iteration begins at `L-2`... stop index is 0, which is the
        # input layer None
        for lyr in range(self.num_layers-2, 0, -1):

            # Lists for tracking errors accumulated for each
            # training example
            dCost_dW_lyrs_samples = []
            dCost_dBias_lyrs_samples = []

            # Arguments that are independent of training samples
            hidden_activation = self.sequential[lyr].activation_function
            w_of_lyr_plus_one = self.sequential[lyr+1].W

            for sample in range(self.batch_size):

                delta_of_lyr_plus_one = dCost_dBias_lyrs[lyr+1][sample]
                z_lyr = weighted_inputs[lyr][sample]
                a_lyr_minus_one = activations[lyr-1][sample]

                # Compute errors
                dCost_dBias_lyr_sample = self._compute_delta_hidden_lyr(
                    wt_matrix_of_lyr_plus_one=w_of_lyr_plus_one,
                    delta_of_lyr_plus_one=delta_of_lyr_plus_one,
                    wted_input_of_cur_lyr=z_lyr,
                    hidden_activation=hidden_activation)

                dCost_dW_lyr_sample = self._compute_deriv_cost_wrt_wt(
                    activations_prev_lyr=a_lyr_minus_one,
                    delta_cur_lyr=dCost_dBias_lyr_sample)

                # Append to samples list
                dCost_dBias_lyrs_samples.append(dCost_dBias_lyr_sample)
                dCost_dW_lyrs_samples.append(dCost_dW_lyr_sample)

            # Update layer list
            dCost_dW_lyrs[lyr] = np.array(dCost_dW_lyrs_samples)
            dCost_dBias_lyrs[lyr] = np.array(dCost_dBias_lyrs_samples)

        # Convert to ndarray
        dCost_dW_lyrs = np.array(dCost_dW_lyrs, dtype=object)
        dCost_dBias_lyrs = np.array(dCost_dBias_lyrs, dtype=object)

        # Debugging and printing shapes
        if self.debug:
            print('MLP._backward_pass')
            print('Length of error tensors... \
                one per layer where first element is None for placeholding input layer')
            print(dCost_dW_lyrs.shape, dCost_dW_lyrs.dtype)
            print(dCost_dBias_lyrs.shape, dCost_dBias_lyrs.dtype)

            print()
            print('Shapes of dC arrays:')
            print('Shapes of dC/dB arrays')

            # Check if all layers are finite
            dCost_dBias_is_finite = np.all([
                np.all(np.isfinite(dcdb))
                for dcdb in dCost_dBias_lyrs if dcdb is not None])

            dCost_dW_is_finite = np.all([
                np.all(np.isfinite(dcdw))
                for dcdw in dCost_dW_lyrs if dcdw is not None])

            if dCost_dBias_is_finite:

                for dcdb in dCost_dBias_lyrs:
                    if dcdb is not None:
                        print(np.array(dcdb).shape)
                    else:
                        print('None')
                    print()
            else:
                print('Not finite....')
                print(dCost_dBias_lyrs)

            print('Shapes of dC/dW arrays')
            if dCost_dW_is_finite:
                for dcdw in dCost_dW_lyrs:
                    if dcdw is not None:
                        print(np.array(dcdw).shape)
                    else:
                        print('None')
                    print()
            else:
                print('Not finite...')
                print(dCost_dW_lyrs)

        breakpoint()

        return dCost_dW_lyrs, dCost_dBias_lyrs

    def _gradient_descent(
            self,
            weight_grads: List[np.ndarray],
            bias_grads: List[np.ndarray]) -> None:
        """Uses gradient to minimize loss.

        Args:
            weight_grads: Tensor of weight gradient vectors
            bias_grads: Tensor of bias gradient vectors.
        """

        # Zipped model and gradients iterator
        grad_iterator = zip(self.sequential, weight_grads, bias_grads)

        # Skip the first element (None entry for input layer)
        next(grad_iterator)

        # Update gradients
        for cnt, (lyr, wt_grad, bias_grad) in enumerate(grad_iterator):

            if self.debug:
                print('_gradient_descent for lyr =', cnt+1)
                print(
                    f'W = {lyr.W.shape} - a * {np.mean(wt_grad, axis=0).shape}')
                print(
                    f'b = {lyr.b.shape} - a * {np.mean(bias_grad, axis=0).shape}')
                breakpoint()

            # Update parameters... does this do reference or copy....
            lyr.W -= self.learning_rate * np.mean(wt_grad, axis=0)
            lyr.b -= self.learning_rate * np.mean(bias_grad, axis=0)

        # Check if update worked... will refer to last layers weights and biases
        if self.debug:
            print('_gradient descent... check if update of weights worked.')
            print(lyr.W == self.sequential[-1].W,
                  lyr.W is self.sequential[-1].W,
                  lyr.W,
                  self.sequential[-1].W)
            print(lyr.b == self.sequential[-1].b,
                  lyr.b is self.sequential[-1].b,
                  lyr.b,
                  self.sequential[-1].b)
            breakpoint()

    def _cache(self, activations: np.ndarray, weighted_inputs: np.ndarray) -> None:
        """Caches activations and weighted inputs from layer for backprop.

        Args:
            activations: Matrix of activations a (m, n_h)
            weighted_inputs: Matrix of weighted inputs z (m, n_h)
        """

        self.activations_cache.append(activations)
        self.weighted_inputs_cache.append(weighted_inputs)

    def _clear_cache(self,) -> None:
        """Sets cache lists to empty."""

        self.activations_cache = []
        self.weighted_inputs_cache = []

    def _compute_delta_last_lyr(
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

        deriv_activation_of_z = self.target_activation.derivative(
            wted_input_of_final_lyr)

        delta_lyr = grad_cost_wrt_activation * deriv_activation_of_z

        return delta_lyr

    def _compute_delta_hidden_lyr(
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

        wted_err = np.dot(np.transpose(
            wt_matrix_of_lyr_plus_one), delta_of_lyr_plus_one)

        deriv_hidden_activation_of_z = hidden_activation.derivative(
            wted_input_of_cur_lyr)

        wted_derived_activation_err = wted_err * deriv_hidden_activation_of_z

        return wted_derived_activation_err

    def _compute_deriv_cost_wrt_wt(
            self,
            activations_prev_lyr: np.ndarray,
            delta_cur_lyr: np.ndarray) -> np.ndarray:
        """Computes derivative of cost fxn with respect to layer weight.

        Uses activation of previous layer and delta of current layer to
        compute dC/dw_jk.

        Args:
            activations_prev_lyr: Activation vector of shape (n_a, ) of previous layer.
            delta_cur_lyr: Error vector of shape (n_d, ).

        Returns:
            Derivative cost w.r.t to weight matrix vector.
        """

        # Must expand dims because vectors here are assumed to be column
        # vectors...
        # (j, 1)
        delta_cur_lyr = np.expand_dims(delta_cur_lyr, axis=-1)

        # (k, 1)
        activations_prev_lyr = np.expand_dims(activations_prev_lyr, axis=-1)

        # Matches weight matrix shape
        # (jk)
        dCost_dWeight_lyr = np.dot(
            delta_cur_lyr, np.transpose(activations_prev_lyr))

        return dCost_dWeight_lyr
