# CSS586_Project

Music related machine learning project

Potential Tasks:

- Music transcription
- Music representation learning and clustering
- Music generation

## Setup

This is a Python 3 project.

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
