# %% [markdown]
# # MIDI MusicNet string quartet model exploration with pypianoroll representation
#
# Alex Kyllo
# May 9, 2021
#
# An exploratory notebook for playing with model architecture ideas.

# %%
%load_ext autoreload
%autoreload 2

import os
# %%
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pypianoroll
from tensorflow.keras import (Model, Sequential, layers, metrics, optimizers,
                              utils)

sys.path.append("..")
from musiclearn import config, processing

# %% [markdown]
# Get all string quartets from MusicNet dataset in shape
# (

# %%
f = "quartets.npy"
if os.path.isfile(f):
    x = np.load(f)
else:
    x = processing.musicnet_quartets_to_numpy(bars_per_phrase=4)
    np.save(f, x)

n_timesteps = x.shape[1]
n_features = x.shape[2]
shape = (n_timesteps, n_features)
print(shape)

# %% [markdown]
#
# We are predicting what notes each instrument is playing, so this is
# either a multiclass classification problem, or perhaps we need an
# architecture with multiple outputs.
#
# This model fails to learn the problem, probably due to sparsity of
# the pianoroll data.

# %%
opt = optimizers.Adam(learning_rate=0.0001)
act = "tanh"
model = Sequential()
model.add(
    layers.LSTM(
        512,
        activation="tanh",
        input_shape=(n_timesteps, n_features),
        return_sequences=True,
    )
)
model.add(layers.LSTM(256, activation=act, return_sequences=True))
model.add(layers.LSTM(256, activation=act, return_sequences=True))
model.add(layers.LSTM(512, activation=act, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(n_features, activation="sigmoid")))
model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[metrics.Precision(), metrics.Recall()],
)
utils.plot_model(model, show_shapes=True)

# %%
history = model.fit(x, x, epochs=200, validation_split=0.2)
history.model.save("quartet_lstm_model")
np.savez(history.history, "quartet_lstm_history.npz")
