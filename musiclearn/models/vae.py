"""models/vae.py
Music Variational Autoencoders
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


def sample_normal(inputs):
    mu, sigma = inputs
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    sample = mu + tf.exp(0.5 * sigma) * epsilon
    return sample


class Sampling(layers.Layer):
    """A layer for sampling from the latent code distribution."""

    def call(self, inputs):
        return sample_normal(inputs)


def loss(y_true, y_pred, beta=1.0):
    """VAE multi-objective cross entropy / KL divergence loss function"""
    reconst_loss = K.sum(keras.losses.sparse_categorical_crossentropy(y_true, y_pred), axis=1)
    diverge_loss = K.sum(keras.losses.kl_divergence(y_true, y_pred), axis=-1)
    return K.mean(reconst_loss + beta * diverge_loss)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """"""
        # https://keras.io/examples/generative/vae/#define-the-vae-as-a-model-with-a-custom-trainstep
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Validation"""
        print("test_step called!")
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=1
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "val_loss": self.total_loss_tracker.result(),
            "val_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "val_kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs, training=training, mask=mask)
        return self.decoder(z)


def one_track_encoder(latent_dim, n_timesteps, n_notes, training=False):
    inputs = layers.Input(shape=(n_timesteps, 1))
    encoder = layers.Embedding(n_notes, 8, input_length=n_timesteps)(inputs)
    encoder = layers.Reshape((n_timesteps, 8))(encoder)
    encoder = layers.LSTM(128, return_sequences=True)(inputs)
    encoder = layers.Dropout(0.2)(encoder, training=training)
    encoder = layers.LSTM(128, return_sequences=False)(encoder)
    mu = layers.Dense(latent_dim, name="mu")(encoder)
    sigma = layers.Dense(latent_dim, name="sigma")(encoder)
    z = Sampling()([mu, sigma])
    model = keras.Model(inputs, [mu, sigma, z], name="encoder")

    return model


def one_track_decoder(latent_dim, n_timesteps, n_notes, training=False):
    inputs = layers.Input(shape=(latent_dim,))
    decoder = layers.RepeatVector(n_timesteps)(inputs)
    decoder = layers.LSTM(128, return_sequences=True)(decoder)
    decoder = layers.Dropout(0.2)(decoder, training=training)
    decoder = layers.LSTM(128, return_sequences=True)(decoder)
    outputs = layers.TimeDistributed(layers.Dense(n_notes, activation="softmax"))(decoder)
    model = keras.Model(inputs, outputs, name="decoder")

    return model


def build_one_track_vae(
    optimizer, latent_dim, embedding_dim, n_timesteps, n_notes, dropout_rate=0.2
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
    optimizer, latent_dim, embedding_dim, n_timesteps, n_tracks, n_notes, dropout_rate=0.2
):
    """Build the multi-track LSTM-VAE."""
    # define encoder model
    inputs = layers.Input(shape=(n_timesteps, n_tracks))
    encoder = layers.Embedding(n_notes, embedding_dim, input_length=n_timesteps)(inputs)
    encoder = layers.Reshape((n_timesteps, embedding_dim * n_tracks))(encoder)
    encoder = layers.LSTM(256, return_sequences=True)(encoder)
    encoder = layers.Dropout(dropout_rate)(encoder)
    encoder = layers.LSTM(256, return_sequences=False)(encoder)
    mu = layers.Dense(latent_dim, name="mu")(encoder)
    sigma = layers.Dense(latent_dim, name="sigma")(encoder)
    # Latent space sampling
    z = layers.Lambda(sample_normal, output_shape=(latent_dim,))([mu, sigma])
    encoder_model = keras.Model(inputs, [mu, sigma, z])

    # define decoder model
    decoder_input = layers.Input(shape=(latent_dim,))
    decoder = layers.RepeatVector(n_timesteps)(decoder_input)
    decoder = layers.LSTM(256, return_sequences=True)(decoder)
    decoder = layers.Dropout(dropout_rate)(decoder)
    decoder = layers.LSTM(256, return_sequences=True)(decoder)
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
