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

The interpolations are between the following pairs of MusicNet MIDI files:

- `Haydn/2104_op64n5_1.mid` and `Ravel/2178_gr_rqtf2.mid`
- `Haydn/2105_op64n5_2.mid` and `Ravel/2179_gr_rqtf3.mid`
- `Haydn/2106_op64n5_3.mid` and `Ravel/2177_gr_rqtf1.mid`
- `Beethoven/2497_qt11_4.mid` and `Mozart/1822_kv_421_1.mid`
- `Beethoven/2433_qt16_3.mid` and `Mozart/1859_kv_464_2.mid`
- `Beethoven/2368_qt12_4.mid` and `Mozart/1807_kv_387_3.mid`
- `Beethoven/2314_qt15_2.mid` and `Mozart/1791_kv_465_4.mid`
- `Beethoven/2480_qt05_1.mid` and `Mozart/1792_kv_465_1.mid`
- `Beethoven/2481_qt05_2.mid` and `Mozart/1835_kv_590_3.mid`
- `Beethoven/2379_qt08_4.mid` and `Mozart/1805_kv_387_1.mid`
- `Beethoven/2365_qt12_1.mid` and `Mozart/1793_kv_465_2.mid`
- `Beethoven/2562_qt02_4.mid` and `Mozart/1790_kv_465_3.mid`
- `Beethoven/2494_qt11_1.mid` and `Mozart/1789_kv_465_2.mid`
- `Beethoven/2403_qt01_4.mid` and `Mozart/1788_kv_465_1.mid`
- `Beethoven/2376_qt08_1.mid` and `Dvorak/1916_dvq10m1.mid`
- `Beethoven/2384_qt13_4.mid` and `Bach/2242_vs1_2.mid`
- `Beethoven/2560_qt02_2.mid` and `Beethoven/2621_qt07_1.mid`
- `Beethoven/2377_qt08_2.mid` and `Beethoven/2381_qt13_1.mid`

## TODOs:

- [ ] Implement quantitative music generation quality metrics
- [ ] Clean up codebase for submission
