"""config.py
Configuration constants.
Authors: Jack Phan and Alex Kyllo
"""

import os

import dotenv

dotenv.load_dotenv()

MUSICNET_MIDI_DIR = str(os.getenv("MUSICNET_MIDI_DIR"))
FF_MIDI_DIR = str(os.getenv("FF_MIDI_DIR"))
LOGS_DIR = str(os.getenv("LOGS_DIR"))
