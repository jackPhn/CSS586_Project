# %% [markdown]
# # MIDI Model Playground
#
# An exploratory notebook for playing with model architecture ideas.
# %%
%load_ext autoreload
%autoreload 2

# %%
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pypianoroll
from tensorflow.keras import Model, Sequential, layers, optimizers, utils

sys.path.append("..")
from musiclearn import config

# %%
# Picking one song from MusicNet dataset
midi_dir = pathlib.Path(config.MUSICNET_MIDI_DIR)
mid_2494 = midi_dir / "Beethoven" / "2494_qt11_1.mid"
multitrack = pypianoroll.read(mid_2494, resolution=24)

# %% [markdown]
#
# ## Training data representation
#
# Because an entire song is too long music is generally composed in bars or
# multi-bar phrases, we plan to read MIDI files and segment them into phrases,
# one training example per phrase.

# %%
def get_num_beats(multitrack, resolution):
    return len(multitrack.downbeat) // resolution


def get_bar_bounds(bar_index, num_bars, beats_per_bar, resolution):
    start = bar_index * resolution
    end = start + (num_bars * beats_per_bar) * resolution
    return (start, end)


# %%
def bars(multitrack, start_index, num_bars, beats_per_bar, resolution):
    start, end = get_bar_bounds(start_index, num_bars, beats_per_bar, resolution)
    tracks = []
    for track in multitrack.tracks:
        tracks.append(
            pypianoroll.Track(
                name=track.name,
                program=track.program,
                pianoroll=track[start:end],
            )
        )
    return pypianoroll.Multitrack(tracks=tracks, resolution=resolution)


# %% [markdown]
# Plot a bar of Beethoven:

# %%
resolution = 24
first_bar = bars(multitrack, 0, 1, 4, resolution)
# %%
viola_track = first_bar.tracks[2]

# %% [markdown]
#
# ## Pianoroll sequence autoencoder model
#
# Testing out training a simple autoencoder model on a single track.

# %%
# Prepare training examples from the binarized cello track
resolution = 24
beats_per_bar = 4
bars_per_phrase = 2
total_bars = get_num_beats(multitrack, resolution) // beats_per_bar
total_phrases = total_bars // bars_per_phrase

# %%
viola_clips = [
    bars(
        multitrack,
        i * bars_per_phrase,
        bars_per_phrase,
        beats_per_bar,
        resolution,
    )
    .tracks[2]
    .binarize()
    .pianoroll.astype(int)
    for i in range(total_phrases)
]
np_viola = np.array(viola_clips)

# %%
np_viola

# %% [markdown]
#
# The viola only plays notes 48-70, notes outside this range are all zeroes
# We can trim the note space to reduce dimensionality and avoid the model trying
# to fit notes outside the viola's range.
# %%
np_viola.sum(axis=(0, 1))

# %%
np_viola = np_viola[:, :, 48:70]
np_viola.sum(axis=(0, 1))

# %%
# make the features ordinal
# np_viola = np.argmax(np_viola, axis=2).astype(float)
# np_viola = np_viola.reshape(np_viola.shape[0], np_viola.shape[1], 1)

# %%
np_viola

# %%
n_timesteps = np_viola.shape[1]
n_features = np_viola.shape[2]
shape = (n_timesteps, n_features)
print(shape)

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
model.add(layers.LSTM(128, activation=act, return_sequences=True))
model.add(layers.LSTM(128, activation=act, return_sequences=True))
model.add(layers.LSTM(512, activation=act, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(n_features, activation="softmax")))
model.compile(
    optimizer=opt,
    # loss="categorical_crossentropy",
    loss="kullback_leibler_divergence",
    metrics=["categorical_accuracy"],
)
# utils.plot_model(model, show_shapes=True)

# %%
history = model.fit(np_viola, np_viola, epochs=500)

# %%
yhat = model.predict(np_viola)
yhat[0, :, 0]
preds = (yhat > 0.5).astype(int)
# %%
np.argmax(preds[0, :, :], axis=1)
# %%
np.argmax(np_viola[0, :, :], axis=1)

# %%
