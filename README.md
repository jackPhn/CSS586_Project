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

## Data

This project uses the MusicNet reference MIDI files, which can be downloaded here:
[musicnet_midis.tar.gz](https://homes.cs.washington.edu/~thickstn/media/musicnet_midis.tar.gz)

Data file paths and other constants should be specified in a `.env` file in the
project root (this directory). The
[python-dotenv](https://pypi.org/project/python-dotenv/) package is used to read
these into shell environment variables.

You will need to create your own .env file. Below are the contents of
my .env file, yours will have different directory paths depending
where you placed the downloaded music data. Download and unzip the
musicnet_midis file and add `MUSICNET_MIDI_DIR=<path to musicnet_midis directory>`
to the .env file, like this:

```
MUSICNET_MIDI_DIR=/media/hdd1/data/school/css586/musicnet_midis
```

For a current list of configuration constants, see [musiclearn/config.py](./musiclearn/config.py)

## Results

A collection of samples of the generated MIDI files, converted to mp3
format, is presented at:
[mp3samples (Google Drive Link)](https://drive.google.com/drive/folders/12o04uubXtP8WdI62Xe5fjE6wQgzCPG8n?usp=sharing)

Within this directory, there are two subdirectories:

### jack_piano_prediction

Four output samples, one from each model, are provided.

### alex_string_quartet_interpolation

Each of the 18 directories contains five interpolations between two of
the string quartet pieces in the MusicNet corpus, numbered 0-4.


## TODOs:

- [ ] Implement quantitative music generation quality metrics
- [ ] Clean up codebase for submission
