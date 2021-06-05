import numpy as np
from tensorflow.keras import layers, models

latent_dim = 128
state_size = 128
z = np.zeros((1, latent_dim))

# The latent vector z is passed through a fully connected layer with tanh activation
# to set the initial state of a "conductor" RNN which produces one embedding vector
# per subsequence (measure). First conductor is given zero inputs and its h, c states
# are passed to subsequent conductors.
# Each embedding vector is individually pass through a shared dense layer with tanh activation
# to produce initial states for a final bottom-layer decoder rnn with softmax activation.
# At each step of the bottom level decoder, the current embedding is concatenated with the
# previous output to be used as the input (note: conditioning)
# paraphrased from MusicVAE paper: https://arxiv.org/pdf/1803.05428.pdf
# Explanation: https://medium.com/@musicvaeubcse/musicvae-understanding-of-the-googles-work-for-interpolating-two-music-sequences-621dcbfa307c


decoder_input = layers.Input(shape=(latent_dim,))
decoder = layers.Dense(latent_dim * state_size * 4, activation="tanh")(decoder_input)
decoder, state_h, state_c = layers.LSTM(
    latent_dim, initial_state=[], return_sequences=True, return_state=True
)(decoder)
# TODO
