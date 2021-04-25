# %%
import pathlib
import sys
import note_seq
import music21
import pretty_midi
import pandas as pd

sys.path.append("..")
from musiclearn import config

# %%
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
dir(pretty_2494)

# %%
print(pretty_2494.instruments)
print(pretty_2494.instruments[0].notes[0:50])

# %% [markdown] One representation is a piano roll, which is a numpy 2D array of
# notes over time steps at a given frames per second rate. There are 128 notes
# (0-127) and if the note is playing during that time step it will be 1, else 0.
# This allows chords but assumes a single instrument. So it loses instrument
# information unless you produce separate per-instrument piano rolls. It
# produces a sparse matrix since most notes aren't being played at most
# time steps.

# %%
# returns a (notes, time) dimension numpy array
piano_2494 = pretty_2494.get_piano_roll(fs=10)
# Show as pandas DataFrame where rows are time steps and columns are note values
pd.DataFrame(piano_2494.T)

# %% [markdown]
# ## Using Magenta note_seq package
#
# note_seq is a very powerful but undocumented package created by the Magenta
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

# %%
