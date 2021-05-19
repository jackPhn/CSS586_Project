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

# %%
import os
import pathlib
import sys

import numpy as np
from tensorflow.keras import (Model, Sequential, layers, metrics, optimizers,
                              utils)

sys.path.append("..")
from musiclearn import config, processing
from musiclearn.models import vae

# %% [markdown]
# Get all string quartets from MusicNet dataset

# %%
f = "quartets_music21.npy"
if os.path.isfile(f):
    x = np.load(f, allow_pickle=True)
else:
    # Read all MusicNet string quartets
    scores, _ = processing.musicnet_quartets_to_music21(program_ids=[40, 40, 41, 42])
    # Stack them into one big 2D array
    x = np.vstack([processing.score_to_str_array(score, resolution=12) for score in scores])
    np.save(f, x)

x = processing.split_array(x, beats_per_phrase=16, resolution=12)

# %%
n_timesteps = x.shape[1]
n_features = x.shape[2]
shape = (n_timesteps, n_features)
print(shape)

# %%
n_notes = np.unique(x).shape[0]

# %% [markdown]
#

# %%
model = vae.MultiTrackVAE(
    optimizer=optimizers.Adam(learning_rate=0.001),
    n_timesteps=n_timesteps,
    n_features=n_features,
    n_notes=n_notes,
    embedding_dim=4,
)
utils.plot_model(model.model, show_shapes=True)

# %%
history = model.fit(x, x, epochs=200, validation_split=0.2)
history.model.save("quartet_lstm_model")
np.savez(history.history, "quartet_lstm_history.npz")
