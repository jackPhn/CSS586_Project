"""musiclearn_cli.py
Command line interface for the musiclearn project.
Authors: Alex Kyllo and Jack Phan
"""
import logging
import random

import click
import numpy as np
from tensorflow.random import set_seed

from musiclearn import (plotting, sequential_models, single_note_processing,
                        training)

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
@click.option(
    "--bidirectional/--unidirectional",
    default=False,
    help="Use bidirectional LSTM layer in encoder.",
)
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
    bidirectional,
    patience,
):
    """Run MultiTrackVAE experiment named EXP_NAME with hyperparameter options.
    Author: Alex Kyllo"""
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
        bidirectional,
        patience,
    )


@click.command()
@click.option(
    "--model-type",
    type=click.Choice(["lstm", "bidirect", "att", "wavenet"], case_sensitive=False),
    help="Type of model",
)
@click.option("--data-path", type=click.STRING, help="Path to folder stores dataset")
@click.option("--logs-dir", type=click.STRING, help="Folder that stores training logs")
@click.option(
    "--sequence_length",
    type=click.INT,
    default=100,
    help="Length of the sequences of notes used in training",
)
def fit_sequential(model_type, data_path, logs_dir, sequence_length):
    """Fit a sequential model of choice on the specified dataset.
    Author: Jack Phan"""
    notes = single_note_processing.read_midi(data_path)

    n_vocab = single_note_processing.get_num_unique_notes(notes)

    network_input, network_output = single_note_processing.prepare_sequences(sequence_length, notes)

    if model_type == "lstm":
        model = sequential_models.lstm_model(network_input.shape, n_vocab)
    elif model_type == "bidirect":
        model = sequential_models.bidirectional_lstm_model(network_input.shape, n_vocab)
    elif model_type == "att":
        model = sequential_models.attention_lstm_model(network_input.shape, n_vocab)
    elif model_type == "wavenet":
        model = sequential_models.simplified_wavenet(network_input.shape, n_vocab)
    else:
        raise ValueError("Invalid model type")

    sequential_models.train_model(model, sequence_length, model_type, 10, 64)


@click.command()
@click.argument("output-name", type=click.STRING)
@click.option("--data-path", type=click.STRING, help="Path to folder stores dataset")
@click.option(
    "--model-type",
    type=click.Choice(["lstm", "bidirect", "att", "wavenet"], case_sensitive=False),
    help="Type of model",
)
@click.option("--weights-path", type=click.STRING, help="Path to saved weights of the model")
@click.option("--num-notes", type=click.INT, help="Number of notes to generate")
def generate_music(output_name, data_path, model_type, weights_path, num_notes):
    """Generate a short piece of music with a fixed number of notes.
    Author: Jack Phan"""
    # load the model
    notes = single_note_processing.read_midi(data_path)
    n_vocab = single_note_processing.get_num_unique_notes(notes)
    network_input, _ = single_note_processing.prepare_sequences(
        sequential_models.SEQUENCE_LENGTH, notes
    )
    if model_type == "lstm" or model_type == "bidirect":
        model = sequential_models.load_lstm_model(weights_path)
    elif model_type == "att":
        model = sequential_models.load_attention_lstm_model(
            weights_path, network_input.shape, n_vocab
        )
    elif model_type == "wavenet":
        model = sequential_models.load_wavenet_model(weights_path)
    else:
        raise ValueError("Invalid model type")

    sequential_models.generate_midi_sample(model, data_path, output_name, num_notes)


@click.command()
@click.argument("historyfile", type=click.Path(exists=True))
@click.argument("outputfile", type=click.Path())
def plot_losses(historyfile, outputfile):
    """Plot model training & validation loss curves from HISTORYFILE and save to OUTPUTFILE."""
    plotting.plot_learning_curves(historyfile, outputfile)


@click.group()
def cli():
    """Command line interface for the musiclearn project"""


def main():
    """"""
    cli.add_command(fit_mtvae)
    cli.add_command(fit_sequential)
    cli.add_command(generate_music)
    cli.add_command(plot_losses)
    cli()


if __name__ == "__main__":
    SEED = 1337
    set_seeds(SEED)
    main()
