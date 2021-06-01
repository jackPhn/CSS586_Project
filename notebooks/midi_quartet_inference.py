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
bee_score = processing.midi_to_music21(beethoven_2494)

bee = processing.score_to_array(bee_score, ticks_per_beat)
moz_score = processing.midi_to_music21(mozart_1788)
moz = processing.score_to_array(moz_score, ticks_per_beat)

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
