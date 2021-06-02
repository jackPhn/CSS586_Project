# CSS586_Project

Generative music modeling on symbolic (MIDI) and audio music data.

## Setup

This is a Python 3 project using TensorFlow 2.

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) 3.9 or 3.8 is
recommended.

Dependencies are specified in the conda environment file
[environment.yml](./environment.yml).

To install the dependencies:

```sh
conda env create -f environment.yml
```

To activate the conda virtual environment:

```sh
conda activate musiclearn
```

Data file paths and other constants should be specified in a `.env` file in the
project root (this directory). The
[python-dotenv](https://pypi.org/project/python-dotenv/) package is used to read
these into shell environment variables.

Here's my .env file, yours will have different directory paths depending where
you placed your music data.

```
MUSICNET_DIR=/media/hdd1/data/school/css586/musicnet
MUSICNET_MIDI_DIR=/media/hdd1/data/school/css586/musicnet_midis
MUSICNET_SAMPLE_RATE=44100
```

For a current list of configuration constants, see [musiclearn/config.py](./musiclearn/config.py)

## Remaining TODOs

### MIDI quartet LSTM-VAE model

- [X] Write code to generate interpolations between two pieces and output as MIDI
- [ ] Try bidirectional LSTM layers
- [ ] Experiment with hierarchical "composer" decoder for longer sequences
- [ ] Maybe do a conditional VAE, conditioning on composer name?
- [ ] Run experiments to tune hyperparameters and assess output quality
- [ ] Implement quantitative music generation quality metrics
- [ ] Clean up codebase for submission
