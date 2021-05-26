import os
import dotenv

dotenv.load_dotenv()

MUSICNET_DIR = str(os.getenv("MUSICNET_DIR"))
MUSICNET_MIDI_DIR = str(os.getenv("MUSICNET_MIDI_DIR"))
MUSICNET_SAMPLE_RATE = str(os.getenv("MUSICNET_SAMPLE_RATE"))
FF_MIDI_DIR = str(os.getenv("FF_MIDI_DIR"))
LOGS_DIR=str(os.getenv("LOGS_DIR"))
