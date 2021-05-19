"""models/vae.py
Music Variational Autoencoders
"""
from tensorflow.keras import Model, layers


class MultiTrackVAE:
    """"""

    def __init__(self, optimizer, n_timesteps, n_features, n_notes, embedding_dim):
        self.model = self._build(optimizer, n_timesteps, n_features, n_notes, embedding_dim)

    def _build(self, optimizer, n_timesteps, n_features, n_notes, embedding_dim):
        """Build and compile the network"""
        inputs = layers.Input(shape=(n_timesteps, n_features))
        embedding = layers.Embedding(n_notes, embedding_dim, input_length=n_timesteps)(inputs)
        reshape = layers.Reshape(
            (n_timesteps, n_features * embedding_dim),
            input_shape=(n_timesteps, n_features, embedding_dim),
        )(embedding)
        encoder = layers.LSTM(512, return_sequences=True)(reshape)
        encoder = layers.LSTM(256)(encoder)
        repeater = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(256, return_sequences=True)(repeater)
        decoder = layers.LSTM(512, return_sequences=True)(decoder)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(decoder)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        return model

    def train(self, x, batch_size, epochs, val_split=0.2):
        """Fit the model to data"""
        history = self.model.fit(
            x, x, batch_size=batch_size, epochs=epochs, validation_split=val_split
        )
        return history
