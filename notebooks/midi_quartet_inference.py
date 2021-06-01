# %% [markdown]
# # Quartet model generation and interpolation

# %%
import os
import sys
from pathlib import Path

sys.path.append("..")
import numpy as np
from musiclearn import config, processing
from musiclearn.models import vae

# %%
midis = Path(config.MUSICNET_MIDI_DIR)
beethoven_2494 = midis / "Beethoven" / "2494_qt11_1.mid"
mozart_1788 = midis / "Mozart" / "1788_kv_465_1.mid"

# %%
ticks_per_beat = 4
beats_per_phrase = 4
programs = [40, 40, 41, 42]
bee = processing.score_to_array(processing.midi_to_music21(beethoven_2494), ticks_per_beat)
moz = processing.score_to_array(processing.midi_to_music21(mozart_1788), ticks_per_beat)

# %%
# saved_model_path = "../experiments/mtvae_0009/2021-05-31T20:24:32/saved_model"
# saved_model_path = "../experiments/mtvae_0013/2021-05-31T22:15:37/saved_model"
# saved_model_path = "../experiments/mtvae_0015/2021-05-31T22:38:59/saved_model"
saved_model_path = "../experiments/mtvae_0016/2021-05-31T23:01:43/saved_model"
model = vae.MultiTrackVAE.from_saved(saved_model_path)

# %% [markdown]
# ## Reconstruction
#
# Use the model to reconstruct an entire piece of music

# %%
bee_reconst = model.reconstruct(bee, ticks_per_beat, beats_per_phrase)
bee_reconst_score = processing.array_to_score(
    bee_reconst, programs=programs, resolution=ticks_per_beat
)

bee_reconst_score.write("midi", "../outputs/2494_qt11_1_reconst_0016.mid")

# %%
moz_reconst = model.reconstruct(moz, ticks_per_beat, beats_per_phrase)
moz_reconst_score = processing.array_to_score(
    moz_reconst, programs=programs, resolution=ticks_per_beat
)
moz_reconst_score.write("midi", "../outputs/1788_kv_465_1_reconst_0016.mid")

# %% [markdown]
# ## Interpolation
#
# Use the model to blend two pieces of music by taking the mean of their
# latent codes

# %%
start = bee
stop = moz
beemoz = model.interpolate(start, stop, 3, ticks_per_beat, beats_per_phrase)
beemoz_score = processing.array_to_score(beemoz[1], programs=programs, resolution=ticks_per_beat)
beemoz_score.write("midi", "../outputs/beemoz_0016.mid")

# %%
start = model.ord_enc.transform(start).astype(int)
stop = model.ord_enc.transform(stop).astype(int)
start = processing.split_array(
    start, beats_per_phrase=beats_per_phrase, resolution=ticks_per_beat, fill=model.rest_code
)
stop = processing.split_array(
    stop, beats_per_phrase=beats_per_phrase, resolution=ticks_per_beat, fill=model.rest_code
)
min_len = min(start.shape[0], stop.shape[0])
start = start[0:min_len, :, :]
stop = stop[0:min_len, :, :]
start_mu, start_sigma, start_z = model.encoder_model.predict(start)
stop_mu, stop_sigma, stop_z = model.encoder_model.predict(stop)
space = np.linspace(start_z, stop_z, 3)
interpolations = []
for x in space:
    x = model.decoder_model.predict(x)
    x = np.stack(x, axis=2)
    x = np.argmax(x, axis=3)
    x = np.vstack(x)
    x = model.ord_enc.inverse_transform(x)
    interpolations.append(x)
