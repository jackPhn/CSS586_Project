"""models/vae.py
Music Variational Autoencoders
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Model:
    def __init__(self, *args):
        self.model = self._build(*args)

    def train(self, x, y, batch_size, epochs, val_split=0.2):
        """Fit the model to data"""
        history = self.model.fit(
            x, y, batch_size=batch_size, epochs=epochs, validation_split=val_split
        )
        return history


def sample_normal(inputs):
    mu, sigma = inputs
    batch = keras.backend.shape(mu)[0]
    dim = keras.backend.int_shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    sample = mu + tf.exp(0.5 * sigma) * epsilon
    return sample


class Sampling(layers.Layer):
    """A layer for sampling from the latent code distribution."""

    def call(self, inputs):
        return sample_normal(inputs)


def loss(y_true, y_pred, beta=1.0):
    """VAE multi-objective cross entropy / KL divergence loss function"""
    reconst_loss = keras.backend.sum(
        keras.losses.sparse_categorical_crossentropy(y_true, y_pred), axis=1
    )
    diverge_loss = keras.backend.sum(keras.losses.kl_divergence(y_true, y_pred), axis=-1)
    return keras.backend.mean(reconst_loss + beta * diverge_loss)


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
        # based on https://keras.io/examples/generative/vae/#define-the-vae-as-a-model-with-a-custom-trainstep
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

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs, training=training, mask=mask)
        return self.decoder(z)


def one_track_encoder(latent_dim, n_timesteps, n_notes, training=False):
    inputs = layers.Input(shape=(n_timesteps, 1))
    x = layers.Embedding(n_notes, 8, input_length=n_timesteps)(inputs)
    x = layers.Reshape((n_timesteps, 8))(x)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x, training=training)
    x = layers.LSTM(128, return_sequences=False)(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    sigma = layers.Dense(latent_dim, name="sigma")(x)
    z = Sampling()([mu, sigma])
    encoder = keras.Model(inputs, [mu, sigma, z], name="encoder")

    return encoder


def one_track_decoder(latent_dim, n_timesteps, n_notes, training=False):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(n_timesteps)(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x, training=training)
    x = layers.LSTM(128, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_notes, activation="softmax"))(x)
    decoder = keras.Model(inputs, outputs, name="decoder")

    return decoder


class OneTrackAE(Model):
    def __init__(self, optimizer, n_timesteps, n_notes):
        self.model = self._build(optimizer, n_timesteps, n_notes)

    def _build(self, optimizer, n_timesteps, n_notes):
        """Build and compile the network"""
        inputs = layers.Input(shape=(n_timesteps, 1))
        encoder = layers.Embedding(n_notes, 8, input_length=n_timesteps)(inputs)
        encoder = layers.Reshape((n_timesteps, 8))(encoder)
        encoder = layers.LSTM(128, return_sequences=True)(inputs)
        encoder = layers.Dropout(0.2)(encoder)
        encoder = layers.LSTM(128, return_sequences=False)(encoder)
        decoder = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(128, return_sequences=True)(decoder)
        decoder = layers.Dropout(0.2)(decoder)
        decoder = layers.LSTM(128, return_sequences=True)(decoder)
        decoder = layers.TimeDistributed(layers.Dense(n_notes, activation="softmax"))(decoder)
        model = Model(inputs=inputs, outputs=decoder)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        return model


class MultiTrackVAE(Model):
    """"""

    # TODO
    def __init__(self, optimizer, n_timesteps, n_features, n_notes):
        super().__init__(optimizer, n_timesteps, n_features, n_notes)

    def _build(self, optimizer, n_timesteps, n_features, n_notes):
        """Build and compile the network"""
        inputs = layers.Input(shape=(n_timesteps, n_features))
        encoder = layers.LSTM(512, return_sequences=True)(inputs)
        decoder = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(512, return_sequences=True)(decoder)
        decoder = layers.TimeDistributed(layers.Dense(n_features * n_notes))(decoder)
        outputs = layers.Dense(n_features, activation="softmax")(decoder)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        return model
