"""plotting.py
Functions for plotting results.
Author: Alex Kyllo
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from musiclearn.processing import REST, SUSTAIN


def plot_learning_curves(history_file: os.PathLike, dest_path: os.PathLike):
    """Plot the train vs. validation learning curves from a history file."""
    if dest_path.endswith(".pgf"):
        matplotlib.use("pgf")
    df = pd.read_csv(history_file)
    fig, ax = plt.subplots()
    y = df.loss
    y_val = df.val_loss
    label = "Loss"
    ax.plot(df.epoch, y, label=f"Training {label}")
    ax.plot(df.epoch, y_val, label=f"Validation {label}")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel(f"{label} Score")
    ax.set_ylim(0.0)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    plt.savefig(
        dest_path,
    )

    return (fig, ax)


def plot_pitch_class_histogram(x: np.array, dest_path: os.PathLike, by_instrument: bool = True):
    """Plot a histogram of pitches"""
    if dest_path.endswith(".pgf"):
        matplotlib.use("pgf")
    xp = np.where(np.isin(x, [REST, SUSTAIN]), np.nan, x)
    tracks = ["violin 1", "violin 2", "viola", "cello"]
    df = pd.DataFrame(xp, columns=tracks)
    ax = sns.histplot(data=df)
    ax.set_xlabel("MIDI pitch value")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    plt.savefig(dest_path)
    return ax
