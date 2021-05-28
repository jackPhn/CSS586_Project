"""models/vae.py
Music Variational Autoencoders
"""
import itertools

import numpy as np
import tensorflow as tf
from musiclearn import processing
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers


def sample_normal(inputs):
    mu, sigma = inputs
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    sample = mu + tf.exp(0.5 * sigma) * epsilon
    return sample


def build_one_track_vae(
    optimizer,
    latent_dim,
    embedding_dim,
    n_timesteps,
    n_notes,
    dropout_rate=0.2,
):
    """Build the one-track LSTM-VAE"""
    # define encoder model
    inputs = layers.Input(shape=(n_timesteps, 1))
    encoder = layers.Embedding(n_notes, embedding_dim, input_length=n_timesteps)(inputs)
    encoder = layers.Reshape((n_timesteps, embedding_dim))(encoder)
    encoder = layers.LSTM(256, return_sequences=True)(encoder)
    encoder = layers.Dropout(dropout_rate)(encoder)
    encoder = layers.LSTM(256, return_sequences=False)(encoder)
    mu = layers.Dense(latent_dim, name="mu")(encoder)
    sigma = layers.Dense(latent_dim, name="sigma")(encoder)

    z = layers.Lambda(sample_normal, output_shape=(latent_dim,))([mu, sigma])
    encoder_model = keras.Model(inputs, [mu, sigma, z])

    # define decoder model
    decoder_input = layers.Input(shape=(latent_dim,))
    decoder = layers.RepeatVector(n_timesteps)(decoder_input)
    decoder = layers.LSTM(256, return_sequences=True)(decoder)
    decoder = layers.Dropout(dropout_rate)(decoder)
    decoder = layers.LSTM(256, return_sequences=True)(decoder)
    outputs = layers.TimeDistributed(layers.Dense(n_notes, activation="softmax"))(decoder)
    decoder_model = keras.Model(decoder_input, outputs)

    # connect encoder and decoder together
    decoder_outputs = decoder_model(z)
    vae_model = keras.Model(inputs=inputs, outputs=decoder_outputs)

    kl_loss = -0.5 * tf.reduce_mean(sigma - tf.square(mu) - tf.exp(sigma) + 1)
    vae_model.add_loss(kl_loss)
    vae_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return vae_model, encoder_model, decoder_model


def build_multi_track_vae(
    optimizer,
    lstm_units,
    latent_dim,
    embedding_dim,
    n_timesteps,
    n_tracks,
    n_notes,
    dropout_rate=0.2,
    gru=False,
):
    """Build the multi-track LSTM-VAE."""
    # define encoder model
    inputs = layers.Input(shape=(n_timesteps, n_tracks))
    if gru:
        rnn = layers.GRU
    else:
        rnn = layers.LSTM
    if embedding_dim > 0:
        encoder = layers.Embedding(n_notes, embedding_dim, input_length=n_timesteps)(inputs)
        encoder = layers.Reshape((n_timesteps, embedding_dim * n_tracks))(encoder)
        encoder = rnn(lstm_units, return_sequences=True)(encoder)
    else:
        encoder = rnn(lstm_units, return_sequences=True)(inputs)
    encoder = layers.Dropout(dropout_rate)(encoder)
    encoder = rnn(lstm_units, return_sequences=False)(encoder)
    mu = layers.Dense(latent_dim, name="mu")(encoder)
    sigma = layers.Dense(latent_dim, name="sigma")(encoder)
    # Latent space sampling
    z = layers.Lambda(sample_normal, output_shape=(latent_dim,))([mu, sigma])
    encoder_model = keras.Model(inputs, [mu, sigma, z])

    # define decoder model
    decoder_input = layers.Input(shape=(latent_dim,))
    decoder = layers.RepeatVector(n_timesteps)(decoder_input)
    decoder = rnn(lstm_units, return_sequences=True)(decoder)
    decoder = layers.Dropout(dropout_rate)(decoder)
    decoder = rnn(lstm_units, return_sequences=True)(decoder)
    outputs = [
        layers.TimeDistributed(layers.Dense(n_notes, activation="softmax"))(decoder)
        for _ in range(n_tracks)
    ]
    concat_outputs = layers.Concatenate(axis=2)(outputs)
    decoder_model = keras.Model(decoder_input, concat_outputs)

    # connect encoder and decoder together
    decoder_outputs = decoder_model(z)
    vae_model = keras.Model(inputs=inputs, outputs=decoder_outputs)

    kl_loss = -0.5 * tf.reduce_mean(sigma - tf.square(mu) - tf.exp(sigma) + 1)
    vae_model.add_loss(kl_loss)

    vae_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return vae_model, encoder_model, decoder_model


class MultiTrackVAE:
    """A Multi Track LSTM Variational Autoencoder."""

    def __init__(self, lstm_units, embedding_dim, latent_dim, learning_rate, dropout_rate, gru):
        self.lstm_units = lstm_units
        self.n_timesteps, self.n_tracks = (None, None)
        self.optimizer = optimizers.Adam(learning_rate)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.gru = gru
        self.vae_model, self.encoder_model, self.decoder_model = (None, None, None)
        self.ord_enc = None
        self.rest_code = None

    def load(self, filepath):
        """Load the model from a Keras saved model path."""
        self.vae_model = models.load_model(filepath)

        return self

    def load_weights(self, filepath, input_shape, n_notes):
        """Load the model weights from a Keras saved weights file."""
        self.vae_model, self.encoder_model, self.decoder_model = build_multi_track_vae(
            self.optimizer,
            self.lstm_units,
            self.latent_dim,
            self.embedding_dim,
            self.n_timesteps,
            self.n_tracks,
            n_notes,
            self.dropout_rate,
            self.gru,
        )
        self.vae_model.load_weights(filepath)
        return self

    def train(
        self, x, ticks_per_beat, beats_per_phrase, epochs, batch_size, learning_rate, callbacks=None
    ):
        """Train the model on a dataset."""
        # Dataset prep, ordinal encoding
        notes = np.unique(x)
        n_notes = notes.shape[0]
        self.n_tracks = x.shape[1]
        self.ord_enc = OrdinalEncoder(categories=list(itertools.repeat(notes, self.n_tracks)))
        x = self.ord_enc.fit_transform(x).astype(int)
        self.rest_code = np.argwhere(self.ord_enc.categories_[0] == processing.REST)[0][0]
        # Split songs into phrases
        x = processing.split_array(
            x, beats_per_phrase=beats_per_phrase, resolution=ticks_per_beat, fill=self.rest_code
        )
        self.n_timesteps = x.shape[1]
        # Remove phrases that are only rests
        all_rests = self.rest_code * self.n_timesteps * self.n_tracks
        x = x[x.sum(axis=(1, 2)) != all_rests]
        self.vae_model, self.encoder_model, self.decoder_model = build_multi_track_vae(
            self.optimizer,
            self.lstm_units,
            self.latent_dim,
            self.embedding_dim,
            self.n_timesteps,
            self.n_tracks,
            n_notes,
            self.dropout_rate,
        )
        self.history = self.vae_model.fit(
            x,
            tf.unstack(x, axis=2),
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
        )
        return self

    def interpolate(self, start, stop, n):
        """Interpolate n samples from the latent space between two inputs."""
        space = np.linspace(start, stop, n)
        axis = space[:, np.newaxis]
        new_points = self.decoder_model.predict(axis)
        return new_points

    def generate():
        """TODO: write a function to generate MIDI output"""

        raise NotImplementedError()
