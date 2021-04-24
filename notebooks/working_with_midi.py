# %%
import os
import pathlib
import sys

sys.path.append("..")
import note_seq
import music21
import dotenv

dotenv.load_dotenv()

# %% [markdown]
# ## Using Magenta note_seq package

# %%
midi_dir = pathlib.Path(os.getenv("MUSICNET_MIDI_DIR"))
mid_2494 = midi_dir / "Beethoven" / "2494_qt11_1.mid"

# %%
notes_2494 = note_seq.midi_io.midi_file_to_note_sequence(mid_2494)
print(notes_2494.instrument_infos)
note_seq.plot_sequence(notes_2494)

# %% [markdown]
# ## Using Music21 package
parse_2494 = music21.converter.parse(mid_2494)
parts = music21.instrument.partitionByInstrument(parse_2494)
print(parts)
print(list(parts.parts[0])[1:20])
