"""models/vae.py
Music Variational Autoencoders
"""
from tensorflow.keras import Model, layers


class MultiTrackVAE:
    """"""

    def __init__(self):
        pass

    def build(self, optimizer, loss, metrics, n_timesteps, n_features, n_notes, embedding_length):
        """Build and compile the network"""
        inputs = layers.Input(shape=(n_timesteps, n_features))
        embedding = layers.Embedding(n_notes, embedding_length)(inputs)
        encoder = layers.LSTM(512, return_sequences=True)(embedding)
        encoder = layers.LSTM(256)(encoder)
        repeater = layers.RepeatVector()(encoder)
        decoder = layers.LSTM(256, return_sequences=True)(repeater)
        decoder = layers.LSTM(512, return_sequences=True)(decoder)
        decoder = layers.Dense(n_features)(decoder)
        outputs = layers.TimeDistributed()(decoder)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = model

    def train(self, x, epochs, val_split=0.2):
        """Fit the model to data"""
        history = self.model.fit(x, validation_split=val_split)
        return history
