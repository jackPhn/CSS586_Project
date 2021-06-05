"""training.py

Helper functions for running experiments.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from tensorflow.keras import callbacks

from musiclearn import processing, vae_models

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def build_callbacks(exp_path: os.PathLike, patience: int = 10):
    """Build an array of callbacks for model training."""
    os.makedirs(exp_path, exist_ok=True)
    cbacks = [
        callbacks.ModelCheckpoint(
            exp_path / "checkpoints",
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=1,
            restore_best_weights=True,
        ),
        callbacks.CSVLogger(exp_path / "history.csv", separator=",", append=True),
        callbacks.TensorBoard(log_dir=exp_path / "tensorboard", histogram_freq=1),
    ]

    return cbacks


def log_start(exp_name, model_name, **hparams):
    LOG.info(f"Running experiment {exp_name} for model {model_name} with hparams: {hparams}")


def log_end(exp_name, model_name, start_time):
    end_time = datetime.now()
    duration = end_time - start_time
    LOG.info(f"Experiment {exp_name} for model {model_name} finished in {duration}")


def train_mtvae(
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
    patience=10,
):
    """Train the MultiTrack VAE once with a given set of hyperparameters."""
    # String quartet MIDI programs
    x = processing.get_string_quartets(ticks_per_beat)
    mtvae = vae_models.MultiTrackVAE(
        lstm_units,
        embedding_dim,
        latent_dim,
        batch_size,
        learning_rate,
        dropout_rate,
        gru,
        bidirectional,
    )
    model_name = type(mtvae).__name__
    start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    exp_path = Path(f"experiments/{exp_name}/{start_str}")
    cbacks = build_callbacks(exp_path, patience=patience)
    log_start(
        exp_name,
        model_name,
        ticks_per_beat=ticks_per_beat,
        beats_per_phrase=beats_per_phrase,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lstm_units=lstm_units,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        patience=patience,
        gru=gru,
    )
    mtvae.train(x, ticks_per_beat, beats_per_phrase, epochs, callbacks=cbacks)
    log_end(exp_name, model_name, start_time)
    mtvae.save(exp_path / "saved_model")
    LOG.info(f"Model configuration saved to {exp_path / 'saved_model'}")
    return mtvae
