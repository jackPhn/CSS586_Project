"""plotting.py
Functions for plotting results.
Author: Alex Kyllo
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


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
    plt.savefig(
        dest_path,
    )
    return (fig, ax)
