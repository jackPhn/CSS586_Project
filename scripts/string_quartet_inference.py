# %% [markdown]
# # String quartet MTVAE model for interpolation
#
# Author: Alex Kyllo
#
# Date: 2021-06-06

# %% [markdown]
# This notebook demonstrates string quartet music interpolation
# with a trained Multi-Track Variational Autoencoder (MTVAE) model.

# %%
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append("..")
from musiclearn import config, processing, vae_models

# %% [markdown]
# Check the `musicnet_midis` directory path
# and show the 36 string quartet MIDI filenames:

# %%
midis = Path(config.MUSICNET_MIDI_DIR)
str(midis)

# %%
sq = processing.STRING_QUARTETS
sq

# %% [markdown]
# Configure the string quartet MIDI program numbers and
# beat/measure resolution and length

# %%
ticks_per_beat = 4
beats_per_phrase = 4
programs = [40, 40, 41, 42]  # Violin x2, Viola, Cello
num_measures = 16  # length of each sample in measures
num_interpolations = 5  # Number of interpolations per pair

# %% [markdown]
# Load the best model from training experiments

# %%
exp_name = "mtvae"
exp_time = "2021-06-06T09:32:32"
saved_model_path = f"../experiments/{exp_name}/{exp_time}/saved_model"

# %%
model = vae_models.MultiTrackVAE.from_saved(saved_model_path)

# %% [markdown]
# To test the model's ability to reconstruct its inputs and
# interpolate new music between them, we will create 18 pairs
# from the 36 original tracks and process them with the
# model's `interpolate` method.

# %%
# zip the list of filenames together to make pairs
half = len(sq) // 2
left = sq[0:half]
right = sq[half : len(sq)]
pairs = list(zip(left, reversed(right)))
pairs

# %% [markdown]
# ## Interpolation
#
# Now we'll use the model to reconstruct each of the pairs
# and use linear interpolation to generate 3 vectors in between them in the latent space.
#
# We'll truncate each track to the first 16 measures to save time and space.

# %%
output_dir = Path(f"../outputs/{exp_name}")

# %%
for pair in pairs:
    print(f"Interpolating between {pair[0]} and {pair[1]}...")
    pairname = "_".join(
        ["_".join([os.path.dirname(f), os.path.splitext(os.path.basename(f))[0]]) for f in pair]
    )
    pairdir = output_dir / pairname
    os.makedirs(pairdir, exist_ok=True)
    arrays = [
        processing.score_to_array(
            processing.midi_to_music21(midis / f).measures(0, num_measures), ticks_per_beat
        )
        for f in pair
    ]
    interpolations = model.interpolate(*arrays, num_interpolations)
    # write the interpolations to disk as numpy arrays
    npz_path = pairdir / f"interpolations_{num_interpolations}.npz"
    np.savez_compressed(npz_path, *interpolations)
    print(f"NumPy arrays saved to {str(npz_path)}")
    for i, arr in enumerate(interpolations):
        # write the interpolations to disk as MIDI format
        score_i = processing.array_to_score(arr, programs=programs, resolution=ticks_per_beat)
        score_i.write("midi", pairdir / f"interpolation_{num_measures}_{i}.mid")
        score_i.write("musicxml", pairdir / f"interpolation_{num_measures}_{i}.xml")
    print(f"MIDI files saved to {str(pairdir / '*.mid')}")
    print(f"MusicXML files saved to {str(pairdir / '*.xml')}")
    # write the interpolations to disk as MusicXML format (for sheet music printing)

# %% [markdown]
# ## Convert MIDI files to WAV
#
# This way we can listen to them anywhere without synthesizer software.
#
# Requires the [fluidsynth](https://www.fluidsynth.org/) library installed on the system
# and a sound font such as the
# [Fluid Release 3 General-MIDI Soundfont](https://member.keymusician.com/Member/FluidR3_GM/index.html)

# %%
SOUND_FONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
# %%
import midi2audio

fs = midi2audio.FluidSynth(sound_font=SOUND_FONT)

# %%
for pair in pairs:
    pairname = "_".join(
        ["_".join([os.path.dirname(f), os.path.splitext(os.path.basename(f))[0]]) for f in pair]
    )
    pairdir = output_dir / pairname
    mids = pairdir.rglob("*.mid")
    for mid in mids:
        wav = mid.with_suffix(".wav")
        fs.midi_to_audio(mid, wav)
        print(f"{mid} converted to {wav}")

# %% [markdown]
# ## Listen to WAV files
#
# using the IPython Audio widget

# %%
from IPython.display import Audio

# %%
wavs = sorted(list(output_dir.rglob("*.wav")))


# %% [markdown]
# Let's listen to interpolations between:
#
# - Beethoven's String Quartet No 15 in A minor part 2. Allegro ma non tanto
# - Mozart String Quartet No 19 in C major part 4. Allegro molto

# %% [markdown]
# Reconstruction of Beethoven:

# %%
Audio(wavs[0])

# %% [markdown]
# 1/4 of the way from Beethoven to Mozart:

# %%
Audio(wavs[1])

# %% [markdown]
# Halfway between Beethoven and Mozart:

# %%
Audio(wavs[2])

# %% [markdown]
# 3/4 of the way from Beethoven to Mozart:

# %%
Audio(wavs[3])

# %% [markdown]
# Reconstruction of Mozart:

# %%
Audio(wavs[4])
