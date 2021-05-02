"""processing.py
Preprocessing audio data (.wav files) for analysis.
"""
import os
from typing import List
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.io import wavfile
import numpy as np
import pypianoroll
from musiclearn import config


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
    assert os.path.isfile(path)
    mt = pypianoroll.read(str(path))
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
    shapes = [m.tracks[0].pianoroll.shape for m in four_phrases]
    assert len(four_phrases) == 4
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
