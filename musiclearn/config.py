import os
import dotenv

dotenv.load_dotenv()

MUSICNET_DIR = os.getenv("MUSICNET_DIR")
MUSICNET_MIDI_DIR = os.getenv("MUSICNET_MIDI_DIR")
MUSICNET_SAMPLE_RATE = os.getenv("MUSICNET_SAMPLE_RATE")