# %% [markdown]
# # Quartet model generation and interpolation

# %%
import os
import sys
from pathlib import Path

sys.path.append("..")
import numpy as np
from musiclearn import config, processing, vae_models

# %%
midis = Path(config.MUSICNET_MIDI_DIR)
beethoven_2494 = midis / "Beethoven" / "2494_qt11_1.mid"
mozart_1788 = midis / "Mozart" / "1788_kv_465_1.mid"

# %%
ticks_per_beat = 4
beats_per_phrase = 4
programs = [40, 40, 41, 42]  # Violin x2, Viola, Cello

# %%
bee_score = processing.midi_to_music21(beethoven_2494)
bee = processing.score_to_array(bee_score, ticks_per_beat)
moz_score = processing.midi_to_music21(mozart_1788)
moz = processing.score_to_array(moz_score, ticks_per_beat)

# %%
exp_name = "mtvae_0016"
exp_time = "2021-05-31T23:01:43"
saved_model_path = f"../experiments/{exp_name}/{exp_time}/saved_model"
model = vae_models.MultiTrackVAE.from_saved(saved_model_path)

# %% [markdown]
# ## Reconstruction
#
# Use the model to reconstruct an entire piece of music

# %%
bee_reconst = model.reconstruct(bee, ticks_per_beat, beats_per_phrase)
bee_reconst_score = processing.array_to_score(
    bee_reconst, programs=programs, resolution=ticks_per_beat
)

bee_reconst_score.write("midi", f"../outputs/{exp_name}/2494_qt11_1_reconst.mid")

# %%
moz_reconst = model.reconstruct(moz, ticks_per_beat, beats_per_phrase)
moz_reconst_score = processing.array_to_score(
    moz_reconst, programs=programs, resolution=ticks_per_beat
)
moz_reconst_score.write("midi", f"../outputs/{exp_name}/1788_kv_465_1_reconst.mid")

# %% [markdown]
# ## Interpolation
#
# Use the model to blend two pieces of music by taking the mean of their
# latent codes

# %%
# Get first 8 measures of each
bee_8 = processing.score_to_array(bee_score.measures(0, 8), ticks_per_beat)
moz_8 = processing.score_to_array(moz_score.measures(0, 8), ticks_per_beat)

# %%
beemoz = model.interpolate(bee_8, moz_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(beemoz):
    beemoz_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    beemoz_score.write("midi", f"../outputs/{exp_name}/beemoz_8_{i}.mid")
    beemoz_score.write("musicxml", f"../outputs/{exp_name}/beemoz_8_{i}.xml")

# %% [markdown]
# another example: Dvorak and Bach

# %%
dvorak_1916 = midis / "Dvorak" / "1916_dvq10m1.mid"
bach_2242 = midis / "Bach" / "2242_vs1_2.mid"
dvorak_score = processing.midi_to_music21(dvorak_1916)
bach_score = processing.midi_to_music21(bach_2242)
dvorak_8 = processing.score_to_array(dvorak_score.measures(0, 8), ticks_per_beat)
bach_8 = processing.score_to_array(bach_score.measures(0, 8), ticks_per_beat)

# %%
dvorbach = model.interpolate(dvorak_8, bach_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(dvorbach):
    dvorbach_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    dvorbach_score.write("midi", f"../outputs/{exp_name}/dvorbach_8_{i}.mid")

# %% [markdown]
# another example: Beethoven and Bach

# %%
beebach = model.interpolate(bee_8, bach_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(beebach):
    beebach_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    beebach_score.write("midi", f"../outputs/{exp_name}/beebach_8_{i}.mid")

# %% [markdown]
# another example: Beethoven and Dvorak

# %%
beedvor = model.interpolate(bee_8, dvorak_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(beedvor):
    beedvor_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    beedvor_score.write("midi", f"../outputs/{exp_name}/beedvor_8_{i}.mid")


# %% [markdown]
# another example: Haydn and Ravel

haydn_2104 = midis / "Haydn" / "2104_op64n5_1.mid"
ravel_2179 = midis / "Ravel" / "2179_gr_rqtf3.mid"
haydn_score = processing.midi_to_music21(haydn_2104)
ravel_score = processing.midi_to_music21(ravel_2179)
haydn_8 = processing.score_to_array(haydn_score.measures(0, 8), ticks_per_beat)
ravel_8 = processing.score_to_array(ravel_score.measures(0, 8), ticks_per_beat)

# %%
hayvel = model.interpolate(haydn_8, ravel_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(hayvel):
    hayvel_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    hayvel_score.write("midi", f"../outputs/{exp_name}/hayvel_8_{i}.mid")

# %% [markdown]
# another example: Beethoven and Haydn

# %%
beedyn = model.interpolate(bee_8, haydn_8, 5, ticks_per_beat, beats_per_phrase)
for i, score in enumerate(beedyn):
    beedyn_score = processing.array_to_score(score, programs=programs, resolution=ticks_per_beat)
    beedyn_score.write("midi", f"../outputs/{exp_name}/beedyn_8_{i}.mid")
