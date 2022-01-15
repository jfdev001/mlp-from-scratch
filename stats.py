"""Module for plotting and statistical functions."""

from __future__ import annotations
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from xml.dom.pulldom import parseString

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

import scipy.stats as st


class MultiModelHistory:
    """Class for containing histories of multiple models.

    NOTE: The nested dictionary structure itself probably warranted
    its own class because the same methods apply to the conf. interval. dict.
    """

    def __init__(self):
        """Initialize dictionaries for MultiModelHistory."""

        self._nested_dict = dict()
        self._conf_interval_dict = defaultdict(dict)
        self._conf_interval_dict_model_major = None
        self._conf_interval_dict_metric_major = None

    @property
    def model_keys(self,) -> List[str]:
        return list(self._nested_dict.keys())

    @property
    def metric_keys(self,) -> List[str]:
        """Assumes models all have the same metrics."""

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

    @property
    def model_performance(self,) -> Union[List[Dict[str, List[float]]], None]:
        """List of dictionaries for statistical model performance."""

        if not self.is_conf_interval_dict_empty():
            return list(self._conf_interval_dict.values())
        else:
            return None

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

        if model_key not in self.model_keys:
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

        if not self.is_conf_interval_dict_empty():
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
                ci_err = confidence_interval_err(
                    vector=metric_values, alpha=alpha, verbose=verbose)

                # Populate the CI dict with the summary statistic tuple
                Stats = namedtuple('Stats', ['mean', 'error'])
                metric_stats = Stats(mean=mean_metric_values, error=ci_err)
                self._conf_interval_dict[model_keys[ix]][metric] = metric_stats

        # Save a copy of the confidence interval dictionary
        self._conf_interval_dict_model_major = deepcopy(
            self._conf_interval_dict)

    def set_key_to_metric_major(self,) -> None:
        """Reformats the conf. interval dict to metric as parent key."""

        # Make the ci dict metric major if first time being called
        if self._conf_interval_dict_metric_major is None:
            self._conf_interval_dict_metric_major = defaultdict(dict)
            for model_ix, performance in enumerate(self.model_performance):
                for metric_name, metric_stats in performance.items():
                    self._conf_interval_dict_metric_major[metric_name][
                        self.model_keys[model_ix]] = metric_stats

        # Set the main ci dict
        self._conf_interval_dict = deepcopy(
            self._conf_interval_dict_metric_major)

    def set_key_to_model_major(self,) -> None:
        """Reformats the conf. interval dict to model as parent key."""

        self._conf_interval_dict = deepcopy(
            self._conf_interval_dict_model_major)

    def reshape_metrics_to_nkfolds_by_epochs(self, nkfolds: int, epochs: int) -> None:
        """Converts the lists in the nested dict to a matrix."""

        for model_name, metric_dicts in self._nested_dict.items():
            for metric_name, metric_list in metric_dicts.items():
                reshaped_metric_list = np.array(metric_list).reshape(
                    nkfolds, epochs)

                self._nested_dict[model_name][metric_name] = reshaped_metric_list

    def _symmetric_metrics(self,) -> bool:
        """True if all models have the same metrics, False otherwise."""

        raise NotImplementedError

    def is_nested_dict_empty(self,) -> bool:
        """True if nested history dictionary is not populated."""

        return len(list(self._nested_dict.keys())) == 0

    def is_conf_interval_dict_empty(self,) -> bool:
        """True if confidence interfval dictionary is not populated."""

        return len(list(self._conf_interval_dict.keys())) == 0


def confidence_interval_err(
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

    # Cast input ndarray
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    # Validate
    if not(alpha > 0. and alpha < 1,):
        raise ValueError(':param alpha: must be between 0 and 100.')

    # Compute estimators
    mean = np.nanmean(vector)
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

    NOTE: Not in use.
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Return the plot
    return fig


def plot_bar_charts(
        multi_model_history: MultiModelHistory,
        bar_width: float,
        title: str,
        xlabel: str = 'Metric',
        ylabel: str = 'Performance',
        model_xaxis: bool = False,
        alpha=0.95,
        style: str = './report.mplstyle',) -> Figure:
    """Plots bar charts of different metrics and with error bars.

    On the use of multiple bars in bar charts
    https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    Args:
        pass

    Returns:
        pass
    """

    # Check to build confidence interval dictionary
    if multi_model_history.is_conf_interval_dict_empty():
        multi_model_history.build_conf_interval_dict(alpha=alpha)

    # Set plot format
    if model_xaxis:
        # aka {metric1: {model1: stats, model2: stats}, ...}
        # The xtick labels will be model based
        multi_model_history.set_key_to_metric_major()
        xticks = np.arange(len(multi_model_history.model_keys))
        xticklabels = multi_model_history.model_keys
        bar_labels = multi_model_history.metric_keys

    else:
        # aka {model1: {metric1: stats, metric2: stats}, ...}
        # The xtick labels will be metric based
        multi_model_history.set_key_to_model_major()
        xticks = np.arange(len(multi_model_history.metric_keys))
        xticklabels = multi_model_history.metric_keys
        bar_labels = multi_model_history.model_keys

    # MPL set up
    plt.style.use(style)
    fig, ax = plt.subplots()

    # Plotting bar charts
    barcontainers = []
    for ix, performance in enumerate(multi_model_history.model_performance):
        means = [v.mean for v in performance.values()]
        errs = [v.error for v in performance.values()]

        barcontainer = ax.bar(
            xticks + bar_width * ix,
            means,
            yerr=errs,
            ecolor='black',
            width=bar_width,
            alpha=0.5,
            capsize=10,
            label=bar_labels[ix],
            align='edge')

        barcontainers.append(barcontainer)

    # Labeling
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks + bar_width)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.grid(True)

    for barcontainer in barcontainers:
        autolabel_err_bars(
            ax=ax,
            barcontainer=barcontainer,
            ha='left',)

    return fig


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
