"""processing.py
Preprocessing audio data (.wav files) for analysis.
"""
import os
import sys
from functools import reduce
import pandas as pd
from typing import List, Tuple
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import wavfile
import numpy as np
import pypianoroll
from musiclearn import config
from music21 import converter, instrument, note, stream, chord


class WavDataGenerator(keras.utils.Sequence):
    """A Keras Sequence for providing audio (.wav) file data in batches."""

    def __init__(
        self,
        directory: os.PathLike,
        batch_size: int = 32,
        shuffle: bool = True,
        max_len: int = None,
    ):
        """Post-initialization"""
        self.directory = Path(directory)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = list(self.directory.glob("*.wav"))
        self.max_len = max_len
        self.on_epoch_end()

    def __len__(self):
        """The number of batches per epoch"""
        return len(self.filenames) // self.batch_size

    def _get_batch_filenames(self, index):
        batch_start = index * self.batch_size
        batch_end = (index + 1) * self.batch_size
        indices = self.indices[batch_start:batch_end]
        return [self.filenames[i] for i in indices]

    def __getitem__(self, index):
        """Get one batch of data as a numpy array, padded to length of the longest sequence."""
        fnames = self._get_batch_filenames(index)
        return pad_sequences(
            [wavfile.read(f)[1] for f in fnames],
            padding="pre",
            truncating="pre",
            maxlen=self.max_len,
            value=0.0,
            dtype="float32",
        )

    def on_epoch_end(self):
        "Reshuffle indices after each epoch"
        self.indices = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)


def test_batch_filenames():
    """Test that it can create shuffled batches of filenames"""
    dir = Path(config.MUSICNET_DIR) / "train_data"
    wdg = WavDataGenerator(dir, batch_size=2)
    batch = wdg._get_batch_filenames(0)
    assert Path(batch[0]).is_file
    assert Path(batch[1]).is_file
    # test it can shuffle
    wdg.on_epoch_end()
    batch1 = wdg._get_batch_filenames(0)
    assert Path(batch1[0]).is_file
    assert Path(batch1[1]).is_file
    assert batch[0] != batch1[0]


def test_batch_maxlen():
    """Test that it retrieves data up to a max length"""
    dir = Path(config.MUSICNET_DIR) / "train_data"
    # get 1 second of data per file
    sample_rate = int(config.MUSICNET_SAMPLE_RATE)
    batch_size = 2
    wdg = WavDataGenerator(dir, batch_size=batch_size, max_len=sample_rate)
    batch = wdg[0]
    assert batch.shape == (batch_size, sample_rate)


class MIDIDataGenerator(keras.utils.Sequence):
    """Generates multitrack pianoroll tensors from MIDI files."""

    def __init__(self, directory: os.PathLike):
        """constructor"""
        self.directory = directory

    def __getitem__(self, index):
        """[] accessor. Get one batch as a numpy array."""


def midi_to_multitrack(path: os.PathLike, resolution: int) -> pypianoroll.Multitrack:
    """"""
    if os.path.isdir(path):
        raise IsADirectoryError(f"{path} is a directory, expected a file.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is not a file.")
    try:
        mt = pypianoroll.read(str(path))
    except:
        return None
    return mt


def num_beats(multitrack: pypianoroll.Multitrack) -> int:
    """Get the integer number of beats in a Multitrack piano roll"""
    assert multitrack.downbeat is not None
    assert multitrack.resolution is not None
    return len(multitrack.downbeat) // multitrack.resolution


def test_num_beats():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    assert num_beats(mt) == 609


def get_bar_bounds(bar_index, num_bars, beats_per_bar, resolution):
    start = bar_index * resolution
    end = start + (num_bars * beats_per_bar) * resolution
    return (start, end)


def get_phrase(
    multitrack: pypianoroll.Multitrack,
    start_index: int,
    num_bars: int,
    beats_per_bar: int,
):
    """Get the phrase starting at the given index, where 0 is the first
    phrase."""
    start, end = get_bar_bounds(
        start_index, num_bars, beats_per_bar, multitrack.resolution
    )
    tracks = [
        pypianoroll.Track(
            name=track.name, program=track.program, pianoroll=track[start:end]
        )
        for track in multitrack.tracks
    ]

    return pypianoroll.Multitrack(
        tracks=tracks,
        resolution=multitrack.resolution,
        downbeat=multitrack.downbeat[start:end],
        tempo=multitrack.tempo[start:end],
    )


def test_get_phrase():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    first_four_bars = get_phrase(mt, 0, 4, 4)
    assert first_four_bars.tracks[0].pianoroll.shape == (384, 128)


def split_phrases(
    multitrack: pypianoroll.Multitrack, bars_per_phrase: int, beats_per_bar: int
):
    """get an array of equal length multitrack phrases from a single multitrack"""
    # pad partial phrases with zeroes
    multitrack = multitrack.pad_to_multiple(
        bars_per_phrase * beats_per_bar * multitrack.resolution
    )
    total_bars = num_beats(multitrack) // beats_per_bar
    num_phrases = total_bars // bars_per_phrase
    phrases = [
        get_phrase(multitrack, ph * bars_per_phrase, bars_per_phrase, beats_per_bar)
        for ph in range(num_phrases)
    ]
    return phrases


def test_split_phrases():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    first_16_bars = get_phrase(mt, 0, 16, 4)
    four_phrases = split_phrases(first_16_bars, 4, 4)
    assert len(four_phrases) == 4
    # shapes = [m.tracks[0].pianoroll.shape for m in four_phrases]
    # assert shapes == [(384, 128), (384, 128), (384, 128), (384, 128)]


def pianoroll_to_numpy(multitrack: pypianoroll.Multitrack):
    """Convert a pypianoroll Multitrack object to a numpy 3D array
    of shape (tracks x time steps x note values)"""
    return np.array([track.pianoroll for track in multitrack.tracks])


def test_pianoroll_to_numpy():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    first_four_bars = get_phrase(mt, 0, 4, 4)
    np_four_bars = pianoroll_to_numpy(first_four_bars)
    assert np_four_bars.shape == (4, 384, 128)


def list_instruments(path: os.PathLike) -> List[Tuple[str, int, bool]]:
    """List the instrument tracks in a MIDI file"""
    mt = midi_to_multitrack(path, 24)
    if mt is None:
        return []
    return [(track.name, track.program, track.is_drum) for track in mt.tracks]


def list_dir_instruments(path: os.PathLike) -> pd.DataFrame:
    """List the instrument tracks in a directory of MIDI files"""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    path = Path(path)
    mid_files = list(path.glob("*.mid"))
    tracks = []
    for f in mid_files:
        instruments = list_instruments(f)
        tracks.extend([(str(f), *i) for i in instruments])
    return pd.DataFrame(tracks, columns=["path", "instrument", "program_id", "is_drum"])


def list_musicnet_instruments(path: os.PathLike) -> pd.DataFrame:
    """Report the instrument tracks present in a directory of MIDIs as a DataFrame."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    path = Path(path)
    return pd.concat([list_dir_instruments(d) for d in path.glob("*")])


def multitracks_by_instruments(
    path: os.PathLike,
    program_ids: List[int],
    resolution: int,
) -> Tuple[List[pypianoroll.Multitrack], List[str]]:
    """Get a list of multitracks with the specified program IDs (instrument #s) from a directory"""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    path = Path(path)
    multitracks = []
    fnames = []
    mid_files = list(path.glob("*.mid"))
    for f in mid_files:
        try:
            current_mt = pypianoroll.read(f, resolution=resolution)
            if current_mt:
                # sort the MIDI tracks by program # and check if exactly equal to the list
                current_mt.tracks = sorted(current_mt.tracks, key=lambda x: x.program)
                current_programs = [t.program for t in current_mt.tracks]
                if current_programs == sorted(program_ids):
                    multitracks.append(current_mt)
                    fnames.append(str(f))
        except:  # probably a corrupt MIDI file
            print(f"Failed to read {f}", file=sys.stderr)
    return multitracks, fnames


def multitracks_by_instruments_musicnet(
    path: os.PathLike, program_ids: List[int], resolution: int
) -> Tuple[List[pypianoroll.Multitrack], List[str]]:
    """Get a list of musicnet multitracks with the specified program IDs"""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    path = Path(path)
    composers = os.listdir(path)
    multitracks = []
    fnames = []
    for composer in composers:
        composer_tracks, composer_files = multitracks_by_instruments(
            path / composer, program_ids, resolution
        )
        if len(composer_tracks) > 0:
            multitracks.extend(composer_tracks)
            fnames.extend(composer_files)
    return multitracks, fnames


def multitracks_musicnet_quartets(resolution: int = 12) -> List[pypianoroll.Multitrack]:
    """Get a pypianoroll multitrack for each string quartet in the MusicNet dataset."""
    path = Path(config.MUSICNET_MIDI_DIR)
    string_quartet = [40, 40, 41, 42]
    tracks, _ = multitracks_by_instruments_musicnet(path, string_quartet, resolution)
    return tracks


def multitracks_musicnet_quartets(resolution: int = 12) -> List[pypianoroll.Multitrack]:
    """Get a pypianoroll multitrack for each string quartet in the MusicNet dataset."""
    path = Path(config.MUSICNET_MIDI_DIR)
    piano = [1]
    tracks, _ = multitracks_by_instruments_musicnet(path, piano, resolution)
    return tracks


def get_note_ranges(multitrack: pypianoroll.Multitrack):
    """Get the lowest and highest note value per pianoroll track"""
    # Use this later to clip the track note ranges
    track_ranges = [
        pypianoroll.pitch_range_tuple(track.pianoroll) for track in multitrack.tracks
    ]
    return track_ranges


def get_note_ranges_list(mts: List[pypianoroll.Multitrack]):
    """Get note ranges for a list of multitracks"""
    track_ranges = [get_note_ranges(mt) for mt in mts]

    def reducer(l: List[Tuple[int, int]], r: List[Tuple[int, int]]):
        return [
            (int(np.nanmin([l[i][0], r[i][0]])), int(np.nanmax([l[i][1], r[i][1]])))
            for i in range(len(l))
        ]

    return reduce(reducer, track_ranges)


def musicnet_quartets_to_numpy(
    bars_per_phrase: int,
    beats_per_bar: int,
    resolution: int,
) -> np.array:
    """Get all string quartets in the MusicNet dataset as numpy arrays"""
    mts_musicnet = multitracks_musicnet_quartets(resolution)
    phrases = []
    # TODO: check for and filter out tracks that aren't in 4/4 time?
    for mt in mts_musicnet:
        phrases.extend(split_phrases(mt, bars_per_phrase, beats_per_bar))
    phrases = np.array([p.stack() for p in phrases]).astype(int)
    shape = phrases.shape
    # reshape to (samples, timesteps, features)
    phrases = (phrases > 0).argmax(axis=3).reshape(shape[0], shape[2], shape[1])
    return phrases


def musicnet_piano_to_numpy(
    bars_per_phrase: int, beats_per_bar: int, resolution: int
) -> np.array:
    """Get all piano pieces in the MusicNet dataset as numpy arrays"""
    # TODO


def numpy_to_multitrack(
    x: np.array, programs: List[int], resolution: int, tempo: float = 120.0
):
    """Convert numpy pianoroll back to multitrack so we can save as MIDI
    Input shape should be (n_timesteps, n_tracks * 128)
    Parameters
    ----------
    programs: List[int]
        A list of MIDI program numbers, a string quartet is [40, 40, 41, 42]
    """
    n_tracks = x.shape[1] / 128
    n_timesteps = x.shape[0]
    np_tempo = np.repeat(tempo, n_timesteps)
    if len(programs) != n_tracks:
        raise Exception("x dimension 1 must be programs * 128")
    tracks = []
    for i, p in enumerate(programs):
        start = 128 * i
        end = start + 128
        tracks.append(
            pypianoroll.StandardTrack(
                pianoroll=x[:, start:end], program=p
            ).set_nonzeros(100)
        )
    multitrack = pypianoroll.Multitrack(
        tracks=tracks, resolution=resolution, tempo=np_tempo
    )
    return multitrack


def musicnet_quartets_to_music21(program_ids=[40, 40, 41, 42]):
    """Get musicnet string quartets in music21 score format"""
    path = Path(config.MUSICNET_MIDI_DIR)

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    composers = os.listdir(path)
    scores = []
    fnames = []
    for composer in composers:
        mid_files = list((path / composer).glob("*.mid"))
        for f in mid_files:
            try:
                score = converter.parse(f)
                if score:
                    # sort the MIDI tracks by program # and check if exactly equal to the list
                    programs = sorted(
                        [p.getInstrument().midiProgram for p in score.parts]
                    )
                    if programs == sorted(program_ids):
                        scores.append(score)
                        fnames.append(str(f))
            except:  # probably a corrupt MIDI file
                print(f"Failed to read {f}", file=sys.stderr)
    return scores, fnames


def len_score(score: stream.Score, resolution: int = 12) -> int:
    """Get the length of a score in ticks.
    Parameters
    ----------
    score: music21.stream.Score
    resolution: int
        The number of time steps per beat
    """
    return int(score.flat.highestTime) * resolution


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


REST = 128
SUSTAIN = 129


def score_to_array(score: stream.Score, resolution: int = 12) -> np.array:
    """Convert score DataFrame to a numpy array"""
    total_length = len_score(score) + 1
    arr = np.full((total_length, len(score.parts)), REST, dtype=int)
    for track, part in enumerate(score.parts):
        for item in part.flat:
            if isinstance(item, chord.Chord) or isinstance(item, note.Note):
                position = int(item.offset) * resolution
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


def split_array(arr: np.array, beats_per_phrase: int, resolution: int = 12) -> np.array:
    """Split a song array into phrases"""
    phrase_len = beats_per_phrase * resolution
    n_phrases = int(np.ceil(len(arr) / phrase_len))
    padding = (-len(arr)) % n_phrases
    filling = np.full((padding, arr.shape[1]), fill_value=REST, dtype=int)
    arr_split = np.array(np.split(np.concatenate((arr, filling)), n_phrases))
    return arr_split
