"""training.py

Helper functions for running experiments.
"""
import logging
from datetime import datetime

from tensorflow.keras import callbacks

from musiclearn import processing
from musiclearn.models import vae

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def build_callbacks(exp_name: str, start_time: datetime, patience: int = 10):
    """Build an array of callbacks for model training."""
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    cbacks = [
        callbacks.ModelCheckpoint(f"experiments/{exp_name}/{start_str}/"),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=1,
            restore_best_weights=True,
        ),
        callbacks.CSVLogger(
            f"experiments/{exp_name}/{start_str}/history.csv", separator=",", append=True
        ),
        callbacks.TensorBoard(
            log_dir=f"experiments/{exp_name}/{start_str}/tensorboard/", histogram_freq=1
        ),
    ]
    return cbacks


def log_start(exp_name, **hparams):
    LOG.info(f"Running experiment {exp_name} with hparams: {hparams}")


def log_end(exp_name, start_time):
    end_time = datetime.now()
    duration = end_time - start_time
    LOG.info(f"Experiment {exp_name} finished in {duration}")


def train_mtvae(
    exp_name,
    ticks_per_beat,
    beats_per_phrase,
    epochs,
    batch_size,
    learning_rate,
    latent_dim,
    embedding_dim,
    dropout_rate,
    patience=10,
):
    """"""
    # String quartet MIDI programs
    x = processing.get_string_quartets(ticks_per_beat, beats_per_phrase)
    mtvae = vae.MultiTrackVAE(embedding_dim, latent_dim, learning_rate, dropout_rate)
    start_time = datetime.now()
    cbacks = build_callbacks(exp_name, start_time, patience=patience)
    log_start(
        exp_name,
    )
    mtvae.train(
        x, ticks_per_beat, beats_per_phrase, epochs, batch_size, learning_rate, callbacks=cbacks
    )
    log_end(exp_name, start_time)
