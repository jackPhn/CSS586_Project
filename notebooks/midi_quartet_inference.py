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
# beats_per_phrase = 16
beats_per_phrase = 4
programs = [40, 40, 41, 42]
bee = processing.score_to_array(processing.midi_to_music21(beethoven_2494), ticks_per_beat)
moz = processing.score_to_array(processing.midi_to_music21(mozart_1788), ticks_per_beat)

# %%
# saved_model_path = "../experiments/mtvae_0005/2021-05-31T15:28:18/saved_model"
# saved_model_path = "../experiments/mtvae_0005/2021-05-31T16:06:39/saved_model"
# saved_model_path = "../experiments/mtvae_0006/2021-05-31T16:36:10/saved_model"
# saved_model_path = "../experiments/mtvae_0007/2021-05-31T16:59:11/saved_model"
# saved_model_path = "../experiments/mtvae_0008/2021-05-31T17:16:16/saved_model"
saved_model_path = "../experiments/mtvae_0009/2021-05-31T20:24:32/saved_model"
model = vae.MultiTrackVAE.from_saved(saved_model_path)

# %% [markdown]
# Use the model to autoencode / reconstruct an entire piece of music

# %%
bee_reconst = model.reconstruct(bee, ticks_per_beat, beats_per_phrase)
bee_reconst_score = processing.array_to_score(
    bee_reconst, programs=programs, resolution=ticks_per_beat
)

bee_reconst_score.write("midi", "../outputs/2494_qt11_1_reconst.mid")

# %%
moz_reconst = model.reconstruct(moz, ticks_per_beat, beats_per_phrase)
moz_reconst_score = processing.array_to_score(
    moz_reconst, programs=programs, resolution=ticks_per_beat
)
moz_reconst_score.write("midi", "../outputs/1788_kv_465_1_reconst.mid")
