"""Module for plotting and statistical functions."""

from __future__ import annotations
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

import scipy.stats as st


class MultiModelHistory:
    """Class for containing histories of multiple models."""

    def __init__(self):
        """Initialize dictionaries for MultiModelHistory."""

        self._nested_dict = dict()
        self._conf_interval_dict = dict()

    @property
    def model_keys(self,) -> List[str]:
        return list(self._nested_dict.keys())

    @property
    def metric_keys(self,) -> List[str]:
        return list(list(self.model_histories)[0].keys())

    @property
    def model_histories(self,) -> List[Dict[str, List[float]]]:
        return list(self._nested_dict.values())

    @property
    def nested_dict(self,) -> Dict[str, Dict[str, List[float]]]:
        """Returns nested dictionary.

        The format of the dictionary is thus:

        {
            model_name1: {metric_1: [epoch1, epoch2, ..], ...},
            model_name2: ...
        }
        """

        return self._nested_dict

    @property
    def conf_interval_dict(self,) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Returns the confidence interval dictionary.

        The format of the dictionary is thus:
        {
            model_name1: {metric_name1: (mean, conf_err), metric_name2: ...},
            model_name2: ...
        }
        """

        return self._conf_interval_dict

    def append_model_history(
            self, model_history: Dict[str, List[float]], model_key: str) -> None:
        """Append a model history and the desired key name to the nested dict.

        Args:
            model_history: The history dictionary of the model.
            model_key: The name of the model.
        """

        self._nested_dict[model_key] = model_history

    def append_kth_fold_model_history(
            self, model_history: Dict[str, List[float]], model_key: str) -> None:
        """Append model history metrics with the current nested dict metrics.

        Args:
            model_history: The history dictionary of the model.
            model_key: The name of the model.
        """

        if self._is_nested_dict_empty():
            self.append_model_history(
                model_history=model_history, model_key=model_key)
        else:
            for metric_name, metric_values in model_history.items():
                self._nested_dict[model_key][metric_name] += metric_values

    def build_conf_interval_dict(
            self,
            alpha: float = 0.95,
            verbose: bool = False) -> None:
        """Builds the confidence interval dictionary.

        Args:
            alpha: Confidence level for error.
            verbose: Prints whether central limit theorem is assumed.
        """

        if not self._is_conf_interval_dict_empty():
            raise ValueError(
                'Cannot build the confidence interval dict multiple times')

        # Prevents overwriting nested dict model histories
        model_histories = deepcopy(self.model_histories)

        # For less verbosity
        model_keys = self.model_keys

        # Iterate through histories in model
        for ix, history in enumerate(model_histories):

            # Iterate through metrics and values in history
            for metric, metric_values in history.items():

                # Compute summary statistics for a metrics values
                mean_metric_values = np.mean(metric_values)
                ci_err = self.confidence_interval_err(
                    vector=metric_values, alpha=alpha, verbose=verbose)

                # Populate the CI dict with the summary statistic tuple
                stat_tuple = (mean_metric_values, ci_err)
                self._conf_interval_dict[model_keys[ix]][metric] = stat_tuple

    def _symmetric_metrics(self,) -> bool:
        """True if all models have the same metrics, False otherwise."""

        raise NotImplementedError

    def _is_nested_dict_empty(self,) -> bool:
        """True if nested history dictionary is not populated."""

        return len(list(self._nested_dict.keys())) == 0

    def _is_conf_interval_dict_empty(self,) -> bool:
        """True if confidence interfval dictionary is not populated."""

        return len(list(self._conf_interval_dict.keys())) == 0

    def confidence_interval_err(
            self,
            vector: np.ndarray,
            alpha: float = 0.95,
            verbose: bool = False) -> float:
        """Computes desired confidence interval error for figures.

        https://en.wikipedia.org/wiki/Confidence_interval

        Args:
            vector: List of values over which a confidence interval will be computed.
            alpha: Confidence level.
            verbose: Prints whether central limit theorem is assumed.

        Returns:
            The confidence level error
        """

        # Validate
        if not(alpha > 0. and alpha < 1,):
            raise ValueError(':param alpha: must be between 0 and 100.')

        # Compute estimators
        mean = np.mean(vector)
        scale = st.sem(vector)  # standard error mean (how close to pop. mean)

        # Determine central limit theorem assumption
        if vector.shape[0] >= 30:
            if verbose:
                print('Assume CLT...')
            interval = st.norm.interval(alpha=alpha, loc=mean, scale=scale)

        else:
            if verbose:
                print('Not assuming CLT...')
            interval = st.t.interval(alpha=alpha, df=len(
                vector)-1, loc=mean, scale=scale)

        # Confidence intervals are m +- h
        # from left_bound = mean - h and right_bound = mean + h
        # From the 'Example' at https://en.wikipedia.org/wiki/Confidence_interval
        # it is clear that [mean - cs/sqrt(n), mean + cs/sqrt(n)] means that the
        # value of cs/sqrt(n), denoted err can be computed with simple rearrangement.
        left_bound, right_bound = interval
        err = right_bound - mean

        # Resulting err for errorbars
        return err


def plot_train_val_loss(
        history_dictionary: Dict[str, List[float]],
        title: str,
        xlabel: str = 'Epoch',
        ylabel: str = 'Metric',
        scatter: bool = False,
        style: str = './report.mplstyle',) -> Figure:
    """Plots training and validation losses using model history dict.

    Args:
        history_dictionary: A dictionary whose keys are the metrics
            of the model during fitting and whose values are the losses
            per epoch. An epoch consists of (num_samples//batch_size)
            batches and the average of the losses of these batches is
            the loss for an epoch.
        title: Title for figure.
        xlabel: Label of x-axis for figure.
        ylabel: Label of y-axis for figure.
        scatter: True for figure to be scatter plot for each metric, false 
            for line plot.
        style: Path to matplotlib style file.

    Returns:
        A figure for the model performance.
    """

    # MPL setup
    plt.style.use(style)
    fig, ax = plt.subplots()

    # Plot history
    for metric_name, metric_values in history_dictionary.items():
        if scatter:
            ax.scatter(np.arange(1, len(metric_values)+1),
                       metric_values, label=metric_name)
        else:
            ax.plot(metric_values, label=metric_name)

    # Labeling
    ax.legend()

    # Return the plot
    return fig


def plot_bar_charts(
        multi_model_history: MultiModelHistory,
        bar_width: float,
        title: str,
        ylabel: str = 'Metric',
        scatter: bool = False,
        style: str = './report.mplstyle',) -> Figure:
    """Plots bar charts of different metrics and with error bars.

    On the use of multiple bars in bar charts
    https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    Args:
        pass

    Returns:
        pass
    """

    # MPL set up
    plt.style.use(style)
    fig, ax = plt.subplots()

    # Compute the number of indices... there is one index per
    # model
    indices = len(multi_model_history.model_keys)

    # Compute the number of "groups", (i.e., the number of
    # metrics per "model" index
    groups_per_index = len(multi_model_history.metric_keys)

    barcontainers = []
    for model_ix, model_history in enumerate(multi_model_history.model_histories):
        pass


def autolabel_err_bars(
        ax: Axes,
        barcontainer: BarContainer,
        height_multiplier: float = 1.00,
        round_n: int = 2,
        x_mod: str = 'left',
        ha: str = 'left') -> None:
    """Labels the mean of the barcontainer above the container.
    https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    """

    for ix, rect in enumerate(barcontainer):

        # Height for bar
        h = rect.get_height()

        # # Calculate x based on position of bar
        # if ix == 0:
        #     x = rect.get_x(),
        # elif ix == len(barcontainer):
        #     x = rect.get_x() + rect.get_width()/2
        # # else:
        # #     x = rect.get_x() + rect.get_width()

        # Set static x
        if x_mod == 'left':
            x = rect.get_x()
        elif x_mod == 'left_center':
            x = rect.get_x() + rect.get_width()/4
        elif x_mod == 'center':
            x = rect.get_x() + rect.get_width()/2
        elif x_mod == 'right_center':
            x = rect.get_x() + 3 * rect.get_width()/4
        elif x_mod == 'right':
            x = rect.get_x() + rect.get_width()
        elif isinstance(x_mod, float):
            x = rect.get_x() + x_mod
        else:
            raise ValueError(
                'invalid :param x_mod:. Must be in options above or float.')

        # Textbox
        ax.text(
            x=x,
            y=height_multiplier*h,
            s=str(round(h, round_n)),
            ha=ha,
            va='bottom')

    return None
