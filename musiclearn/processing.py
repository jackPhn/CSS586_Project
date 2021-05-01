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


def num_bars(multitrack: pypianoroll.Multitrack) -> int:
    """Get the integer number of bars in a Multitrack piano roll"""
    assert multitrack.downbeat is not None and multitrack.resolution is not None
    return len(multitrack.downbeat) // multitrack.resolution


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

    return pypianoroll.Multitrack(tracks=tracks, resolution=multitrack.resolution)


def get_phrase_array(
    multitrack: pypianoroll.multitrack, bars_per_phrase: int, beats_per_bar: int
) -> np.array:
    """get an array of equal length multitrack phrases from a single multitrack"""
    # TODO


def test_num_bars():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    assert num_bars(mt) == 609


def test_get_phrase():
    path = Path(config.MUSICNET_MIDI_DIR)
    mid_file = list(path.glob("Beethoven/2494*.mid"))[0]
    mt = midi_to_multitrack(mid_file, 24)
    first_four_bars = get_phrase(mt, 0, 4, 4)
    assert first_four_bars.tracks[0].pianoroll.shape == (384, 128)
