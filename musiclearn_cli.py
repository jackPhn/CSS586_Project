"""Command line interface for the musiclearn project."""
import logging
import random

import click
import numpy as np
from tensorflow.random import set_seed

from musiclearn import training

LOG = logging.getLogger("musiclearn")
LOG.setLevel(logging.DEBUG)
fh = logging.FileHandler("musiclearn.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
LOG.addHandler(fh)


def set_seeds(seed):
    """Set random seed for Python, NumPy and TensorFlow"""
    np.random.seed(seed)
    set_seed(seed)
    random.seed(seed)


@click.command()
@click.argument("exp_name", type=click.STRING)
@click.option("--ticks-per-beat", type=click.INT, default=4, help="Time steps per quarter note.")
@click.option("--beats-per-phrase", type=click.INT, default=4, help="Quarter notes per phrase.")
@click.option("--epochs", type=click.INT, default=500, help="The training batch size.")
@click.option("--batch-size", type=click.INT, default=32, help="The training batch size.")
@click.option(
    "--learning-rate", type=click.FLOAT, default=0.0002, help="The optimizer learning rate."
)
@click.option("--lstm-units", type=click.INT, default=256, help="Number of LSTM units per layer.")
@click.option("--latent-dim", type=click.INT, default=100, help="The latent vector dimension.")
@click.option("--embedding-dim", type=click.INT, default=8, help="The note embedding dimension.")
@click.option(
    "--dropout-rate", type=click.FLOAT, default=0.4, help="The dropout rate between LSTM layers"
)
@click.option("--gru/--lstm", default=False, help="Use GRU layer instead of LSTM.")
@click.option("--patience", type=click.INT, default=10, help="The early stopping patience.")
def fit_mtvae(
    exp_name,
    ticks_per_beat,
    beats_per_phrase,
    epochs,
    batch_size,
    learning_rate,
    lstm_units,
    latent_dim,
    embedding_dim,
    dropout_rate,
    gru,
    patience,
):
    """Run MultiTrackVAE experiment named EXP_NAME with hyperparameter options."""
    training.train_mtvae(
        exp_name,
        ticks_per_beat,
        beats_per_phrase,
        epochs,
        batch_size,
        learning_rate,
        lstm_units,
        latent_dim,
        embedding_dim,
        dropout_rate,
        gru,
        patience,
    )


@click.group()
def cli():
    """Command line interface for the musiclearn project"""


def main():
    """"""
    cli.add_command(fit_mtvae)
    cli()


if __name__ == "__main__":
    SEED = 1337
    set_seeds(SEED)
    main()
