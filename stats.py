"""Module for plotting and statistical functions."""

from __future__ import annotations
from typing import Optional, Dict

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

import scipy.stats as st


def plot_train_val_loss(
        history_dictionary: Dict,
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

    plt.style.use(style)

    fig, ax = plt.subplots()
    for metric_name, metric_values in history_dictionary.items():
        if scatter:
            ax.scatter(np.arange(1, len(metric_values)+1),
                       metric_values, label=metric_name)
        else:
            ax.plot(metric_values, label=metric_name)

    ax.legend()

    return fig


def plot_bar_charts():
    """Plots bar charts of different metrics and with error bars.

    Args:
        pass

    Returns:
        pass
    """
    return


def confidence_interval_err(vector: np.ndarray, alpha: float = 0.95) -> float:
    """Computes desired confidence interval error.

    Args:
        pass

    Returns:
        pass
    """

    # Validate
    if not(alpha > 0. and alpha < 1,):
        raise ValueError(':param alpha: must be between 0 and 100.')

    # Compute estimators
    mean = np.mean(vector)
    scale = st.sem(vector)  # standard error mean (how close to pop. mean)

    # Determine central limit theorem assumption
    if vector.shape[0] >= 30:
        print('Assume CLT...')
        interval = st.norm.interval(alpha=alpha, loc=mean, scale=scale)

    else:
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
