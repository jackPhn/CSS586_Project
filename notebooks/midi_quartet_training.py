# %% [Markdown]
# # MIDI MusicNet string quartet model exploration with pypianoroll representation
#
# Alex Kyllo
# May 9, 2021
#
# An exploratory notebook for playing with model architecture ideas.

# %%
import itertools
import os
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras import optimizers

sys.path.append("..")
from musiclearn import processing, vae_models

# %%
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# %% [markdown]
# Get all string quartets from MusicNet dataset

# %%
f = "../data/quartets_music21_4.npy"
program_ids = [40, 40, 41, 42]
ticks_per_beat = 4
if os.path.isfile(f):
    x = np.load(f, allow_pickle=True)
else:
    scores_file = "../data/quartets.joblib"
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
x = processing.split_array(x, beats_per_phrase=4, resolution=ticks_per_beat, fill=REST_CODE)
n_timesteps = x.shape[1]
n_features = x.shape[2]
shape = (n_timesteps, n_features)

# %% [markdown]
# Remove the phrases that consist only of rests

# %%
ALL_RESTS = REST_CODE * n_timesteps * n_features
x = x[x.sum(axis=(1, 2)) != ALL_RESTS]

# %% [markdown]
# Variational LSTM-AE (one track)

# %%
# select only the first instrument
x_0 = x[:, :, 0]
np.random.shuffle(x_0)
# select only the phrases that are not only rests
x_0 = x_0[np.any((x_0 != REST_CODE), axis=1)]
# squash to range 0-1
x_0_in = tf.convert_to_tensor(x_0 / n_notes)

# %%
lstm_units = 256
latent_dim = 100
embedding_dim = 8
opt = optimizers.Adam(learning_rate=0.0002)

# %%
lstm_vae, encoder, decoder = vae_models.build_one_track_vae(
    opt, latent_dim, embedding_dim, n_timesteps, n_notes, dropout_rate=0.2
)
history = lstm_vae.fit(x_0, x_0, batch_size=32, epochs=500, validation_split=0.1)
lstm_vae.save_weights("lstm_vae.hdf5")
# %% [markdown]
# Generate samples from the VAE

pred_0 = np.argmax(lstm_vae.predict(x_0[0:1, :]), axis=2)
pred_0

# %% [markdown]
# Variational LSTM-AE (four track)
mtvae, mencoder, mdecoder = vae_models.build_multi_track_vae(
    opt, lstm_units, latent_dim, embedding_dim, n_timesteps, n_features, n_notes, dropout_rate=0.2
)
mhistory = mtvae.fit(x, tf.unstack(x, axis=2), batch_size=32, epochs=100, validation_split=0.1)
mtvae.save_weights("mtvae.hdf5")
