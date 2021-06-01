# %% [markdown]
# # Exploring MIDI Music Data
#
# An exploratory notebook for reading and manipulating MIDI music data.
# %%
import pathlib
import sys

import matplotlib.pyplot as plt
import music21
import note_seq
import numpy as np
import pandas as pd
import pretty_midi
import pypianoroll
from miditoolkit import midi, pianoroll
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.wrappers import TimeDistributed

sys.path.append("..")
from musiclearn import config

# %%
# Picking one song from MusicNet dataset
midi_dir = pathlib.Path(config.MUSICNET_MIDI_DIR)
mid_2494 = midi_dir / "Beethoven" / "2494_qt11_1.mid"

# %% [markdown]
# ## Using the pretty_midi package
#
# `pretty_midi` encodes the note pitch values, velocities and start and end
# times of each instrument track. Gaps in timing between notes are rests. It
# includes methods for estimating the tempo and time signature, and converting
# between representations.
# %%
pretty_2494 = pretty_midi.PrettyMIDI(str(mid_2494))

# %%
print(pretty_2494.instruments)
print(pretty_2494.instruments[0].notes[0:50])

# %% [markdown]
#
# One representation is a piano roll, which is a numpy 2D array of notes over
# time steps at a given frames per second rate (symbolic timing). There are 128
# notes (0-127) and the values are the note velocities. Or, if binarized, the
# values are 1 for playing, else 0 for resting. This format allows chords but
# assumes a single instrument. So it loses instrument information unless you
# produce separate per-instrument piano rolls. It produces a sparse matrix since
# most notes aren't being played at most time steps.

# %%
# returns a (notes, time) dimension numpy array
piano_2494 = pretty_2494.get_piano_roll(fs=10)
# Show as pandas DataFrame where rows are time steps and columns are note values
pd.DataFrame(piano_2494.T)

# %% [markdown]
# ## Using miditoolkit
#
# miditoolkit is similar to pretty_midi but uses symbolic (tick) timing instead
# of absolute (wall clock) timing. It can also convert to and from pianorolls.
# %%
mid_obj = midi.parser.MidiFile(mid_2494)
notes = mid_obj.instruments[0].notes
print(notes[0:50])
pianoroll_2494 = pianoroll.notes2pianoroll(notes, ticks_per_beat=24)
pd.DataFrame(pianoroll_2494)

# %% [markdown]
# ## Using pypianoroll
#
# Yet another package, `pypianoroll`, can read MIDI into multi-track pianoroll
# sparse tensors, and also provides handy visualizations.

# %%
multitrack = pypianoroll.read(mid_2494, resolution=24)
print(multitrack)
multitrack.plot(mode="separate")
# multitrack.plot(mode="blended")

# %% [markdown]
# ## Using Magenta note_seq package
#
# note_seq is a powerful but undocumented package created by the Magenta
# project. It includes tools for converting between MIDI and a Protocol Buffer
# serializeable representation of symbolic music. It uses pretty_midi internally
# but adds some bells and whistles like plotting.

# %%
notes_2494 = note_seq.midi_io.midi_file_to_note_sequence(mid_2494)
print(notes_2494.instrument_infos)
note_seq.plot_sequence(notes_2494)

# %% [markdown]
# ## Using Music21 package
#
# Another package similar to pretty_midi is Music21. Instead of encoding the
# start and end time of each note, it encodes a metronome in beats per minute,
# and encodes rest symbols. This is closer to a musical score.

# %%
parse_2494 = music21.converter.parse(mid_2494)
parts = music21.instrument.partitionByInstrument(parse_2494)
print(parts)
print(list(parts.parts[0])[1:20])

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
fig, axes = plt.subplots(4, sharex=True, sharey=True)
first_bar = bars(multitrack, 0, 1, 4, resolution)
first_bar.plot(axes)
for ax in axes:
    ax.set_ylim(24, 72)
plt.savefig("../papers/progress/alex/first_bar.png")

# %%
resolution = 24
fig, axes = plt.subplots(4, sharex=True, sharey=True)
first_bar = bars(multitrack, 0, 2, 4, resolution)
first_bar.plot(axes)
for ax in axes:
    ax.set_ylim(24, 72)
plt.savefig("first_two_bars.png")


# %%
viola_track = first_bar.tracks[2]
viola_track.plot()
ax = plt.gca()
ax.set_title("Viola Track")
ax.set_ylim(24, 96)
