"""processing.py
Preprocessing audio data (.wav files) for analysis.
Author: Alex Kyllo
Date: 2021-06-06
"""
import logging
import os
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
from music21 import chord, converter, instrument, note, stream

from musiclearn import config

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# Special note pitch values for rest and sustain
REST = 128
SUSTAIN = 129

# List of all 36 string quartets in the MusicNet dataset
STRING_QUARTETS = [
    "Haydn/2104_op64n5_1.mid",
    "Haydn/2105_op64n5_2.mid",
    "Haydn/2106_op64n5_3.mid",
    "Beethoven/2497_qt11_4.mid",
    "Beethoven/2433_qt16_3.mid",
    "Beethoven/2368_qt12_4.mid",
    "Beethoven/2314_qt15_2.mid",
    "Beethoven/2480_qt05_1.mid",
    "Beethoven/2481_qt05_2.mid",
    "Beethoven/2379_qt08_4.mid",
    "Beethoven/2365_qt12_1.mid",
    "Beethoven/2562_qt02_4.mid",
    "Beethoven/2494_qt11_1.mid",
    "Beethoven/2403_qt01_4.mid",
    "Beethoven/2376_qt08_1.mid",
    "Beethoven/2384_qt13_4.mid",
    "Beethoven/2560_qt02_2.mid",
    "Beethoven/2377_qt08_2.mid",
    "Beethoven/2381_qt13_1.mid",
    "Beethoven/2621_qt07_1.mid",
    "Bach/2242_vs1_2.mid",
    "Dvorak/1916_dvq10m1.mid",
    "Mozart/1788_kv_465_1.mid",
    "Mozart/1789_kv_465_2.mid",
    "Mozart/1790_kv_465_3.mid",
    "Mozart/1793_kv_465_2.mid",
    "Mozart/1805_kv_387_1.mid",
    "Mozart/1835_kv_590_3.mid",
    "Mozart/1792_kv_465_1.mid",
    "Mozart/1791_kv_465_4.mid",
    "Mozart/1807_kv_387_3.mid",
    "Mozart/1859_kv_464_2.mid",
    "Mozart/1822_kv_421_1.mid",
    "Ravel/2177_gr_rqtf1.mid",
    "Ravel/2179_gr_rqtf3.mid",
    "Ravel/2178_gr_rqtf2.mid",
]


def midi_to_music21(f):
    score = converter.parse(f)
    return score


def musicnet_quartets_to_music21(program_ids=None):
    """Get musicnet string quartets in music21 score format"""
    if program_ids is None:
        program_ids = [40, 40, 41, 42]
    path = Path(config.MUSICNET_MIDI_DIR)

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    composers = os.listdir(path)
    scores = []
    fnames = []
    for composer in composers:
        mid_files = list((path / composer).glob("*.mid"))
        LOG.info(f"Reading MIDI files in {str(path / composer)}...")
        for f in mid_files:
            try:
                score = converter.parse(f)
                if score:
                    # sort the MIDI tracks by program # and check if exactly equal to the list
                    programs = sorted([p.getInstrument().midiProgram for p in score.parts])
                    if programs == sorted(program_ids):
                        LOG.info(f"{f} is a match.")
                        scores.append(score)
                        fnames.append(str(f))
            except:  # probably a corrupt MIDI file
                LOG.warn(f"Failed to read {f}")
    return scores, fnames


def len_score(score: stream.Score, resolution: int = 12) -> int:
    """Get the length of a score in ticks.
    Parameters
    ----------
    score: music21.stream.Score
    resolution: int
        The number of time steps per beat
    """
    return int(score.flat.highestTime * resolution)


def score_to_df(score: stream.Score, resolution: int = 12) -> pd.DataFrame:
    """Convert a Music21 score to a pandas dataframe"""
    parts = []

    for track, part in enumerate(score.parts):
        instr = part.getInstrument().midiProgram
        for item in part.flat:
            if isinstance(item, note.Note):
                parts.append(
                    (
                        track,
                        instr,
                        item.offset * resolution,
                        item.quarterLength * resolution,
                        item.pitch.midi,
                    )
                )
            elif isinstance(item, chord.Chord):
                parts.append(
                    (
                        track,
                        instr,
                        item.offset * resolution,
                        item.quarterLength * resolution,
                        item.sortAscending().pitches[0].midi,
                    )
                )
    df = (
        pd.DataFrame(
            parts,
            columns=["track", "instrument", "position", "duration", "pitch"],
            dtype=int,
        )
        .groupby(["track", "instrument", "position"])[["duration", "pitch"]]
        .min()
        .reset_index()
    )

    return df


def score_to_array(score: stream.Score, resolution: int = 12) -> np.array:
    """Convert score to a numpy array"""
    total_length = len_score(score) + 1
    arr = np.full((total_length, len(score.parts)), REST, dtype=int)
    for track, part in enumerate(score.parts):
        for item in part.flat:
            if isinstance(item, chord.Chord) or isinstance(item, note.Note):
                position = int(item.offset * resolution)
                duration = int(item.quarterLength * resolution)
                pitch = (
                    item.pitch.midi
                    if isinstance(item, note.Note)
                    else item.sortAscending().pitches[0].midi
                )
                arr[position, track] = pitch
                for i in range(position + 1, position + duration):
                    arr[i, track] = SUSTAIN

    return arr


def chord_to_str(ch: chord.Chord) -> str:
    """Convert a chord to string representation"""
    return (".").join([p.nameWithOctave for p in ch.sortAscending().pitches])


def str_to_chord(string: str) -> chord.Chord:
    pitches = string.split(".")
    return chord.Chord(pitches)


def array_to_score(
    arr: np.array,
    programs: List[int] = None,
    resolution: int = 12,
    rest: Any = REST,
    sustain: Any = SUSTAIN,
) -> stream.Score:
    """Convert str array back into a score so we can output MIDI.
    Parameters
    ----------
    arr: np.array
        A numpy array of shape (n_timesteps, n_parts)
    programs: List[int]
        A list of MIDI program numbers (instrument codes), one per part
    resolution: int
        The number of time steps per quarter note (beat)
    """
    score = stream.Score()
    num_parts = arr.shape[1]
    df = pd.DataFrame(arr)
    df["offset"] = df.index / resolution
    for p in range(num_parts):
        part = stream.Part()
        mid_program = programs[p] if programs else 0
        inst = instrument.instrumentFromMidiProgram(mid_program)
        part.insert(0, inst)
        dfp = df[[p, "offset"]]
        dfp = dfp[dfp[p] != sustain]
        dfp["duration"] = -dfp["offset"].diff(-1)
        dfp["duration"] = dfp["duration"].fillna(0)
        for i, row in dfp.iterrows():
            if row[p] == rest:
                part.append(note.Rest(quarterLength=row["duration"]))
            elif isinstance(row[p], str) and "." in row[p]:  # chord
                part.append(chord.Chord(str_to_chord(row[p]), quarterLength=row["duration"]))
            else:  # note
                part.append(note.Note(row[p], quarterLength=row["duration"]))

        score.insert(0, part)
    return score


def split_array(arr: np.array, beats_per_phrase: int, resolution: int = 12, fill=REST) -> np.array:
    """Split a song array into phrases."""
    phrase_len = beats_per_phrase * resolution
    n_phrases = int(np.ceil(len(arr) / phrase_len))
    padding = (-len(arr)) % phrase_len
    filling = np.full((padding, arr.shape[1]), fill_value=fill)
    arr_split = np.array(np.split(np.concatenate((arr, filling)), n_phrases))
    return arr_split


def test_split_array():
    """test that split_array works on an example."""
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    score = converter.parse(mid_file)
    arr = score_to_array(score, resolution=12)
    assert arr.shape == (7309, 4)
    arr_split = split_array(arr, beats_per_phrase=16)
    assert arr_split.shape == (39, 192, 4)


def get_string_quartets(ticks_per_beat: int):
    """
    Get all the MusicNet string quartets into a NumPy tensor,
    splitting into phrases.
    """
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    musicnet_midis = Path(config.MUSICNET_MIDI_DIR)
    os.makedirs(here / "../data", exist_ok=True)
    f = here / f"../data/quartets_{ticks_per_beat}.npy"
    if os.path.isfile(f):
        # load processed arrays from saved NumPy file
        LOG.info(f"Reading {str(f)}...")
        x = np.load(f, allow_pickle=True)
    else:
        # Process arrays from saved Music21 scores file
        scores_file = here / "../data/quartets.joblib"
        if os.path.isfile(scores_file):
            scores, fnames = joblib.load(scores_file)
        else:
            # Read all MusicNet string quartets from the raw .mid files
            scores = []
            for sq in STRING_QUARTETS:
                LOG.info(f"Parsing MIDI file {sq}...")
                scores.append(converter.parse(musicnet_midis / sq))
            joblib.dump((scores, STRING_QUARTETS), scores_file)
        # Stack them into one big 2D array
        x = np.vstack([score_to_array(score, resolution=ticks_per_beat) for score in scores])
        np.save(f, x)
    return x


def random_transpose(arr):
    """Pitch shift the array up or down by +/- 6 semitones"""
    shift = np.random.choice(range(1, 7))
    return arr + shift * ~np.isin(arr, [REST, SUSTAIN])


def augment(arr):
    """Random transpose each training example and append to the original dataset."""
    augs = np.vectorize(random_transpose)(arr)
    arr = np.vstack([arr, augs])
    return arr
