# %% [markdown]
# # MIDI MusicNet string quartet model exploration with pypianoroll representation
#
# Alex Kyllo
# May 9, 2021
#
# An exploratory notebook for playing with model architecture ideas.

# %%
#%load_ext autoreload
#%autoreload 2

import itertools
# %%
import os
import pathlib
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers, losses, metrics, optimizers, utils

sys.path.append("..")
from musiclearn import processing
from musiclearn.models import vae

# %% [markdown]
# Get all string quartets from MusicNet dataset

# %%
f = "quartets_music21_nochords.npy"
program_ids = [40, 40, 41, 42]
ticks_per_beat = 4
if os.path.isfile(f):
    x = np.load(f, allow_pickle=True)
else:
    scores_file = "scores.joblib"
    if os.path.isfile(scores_file):
        scores, fnames = joblib.load(scores_file)
    else:
        # Read all MusicNet string quartets
        scores, fnames = processing.musicnet_quartets_to_music21(program_ids=program_ids)
        joblib.dump((scores, fnames), scores_file)
    # Stack them into one big 2D array
    x = np.vstack([processing.score_to_array(score, resolution=ticks_per_beat) for score in scores])
    np.save(f, x)

# %%
notes = np.unique(x)
n_notes = notes.shape[0]
enc = OrdinalEncoder(categories=list(itertools.repeat(notes, 4)))
x = enc.fit_transform(x).astype(int)
REST_CODE = np.argwhere(enc.categories_[0] == processing.REST)[0][0]

# %%
x = processing.split_array(x, beats_per_phrase=8, resolution=ticks_per_beat, fill=REST_CODE)
n_timesteps = x.shape[1]
n_features = x.shape[2]
shape = (n_timesteps, n_features)

# %% [markdown]
# First, one track autoencoder

# %%
model = vae.OneTrackAE(
    optimizer=optimizers.Adam(learning_rate=0.001),
    n_timesteps=n_timesteps,
    n_notes=n_notes,
)
utils.plot_model(model.model, show_shapes=True)

# %%
# select only the first instrument
x_0 = x[:, :, 0]
np.random.shuffle(x_0)
# select only the phrases that are not only rests
x_0 = x_0[np.any((x_0 != REST_CODE), axis=1)]
# squash to range 0-1
x_0_in = x_0 / n_notes
# %%
history = model.train(x_0, x_0, batch_size=32, epochs=100, val_split=0.2)
history.model.save("violin_lstm_model")
np.savez(history.history, "violin_lstm_history.npz")

# %% [markdown]
# Variational AE (one track)

# %%
latent_dim = 8
opt = optimizers.Adam(learning_rate=0.0001)
encoder = vae.one_track_encoder(latent_dim, n_timesteps, n_notes)
decoder = vae.one_track_decoder(latent_dim, n_timesteps, n_notes)
lstm_vae = vae.VAE(encoder, decoder)
lstm_vae.compile(optimizer=opt)
history = lstm_vae.fit(x_0, batch_size=32, epochs=100, validation_split=0.1)

# TODO: implement VAE for multi-track
