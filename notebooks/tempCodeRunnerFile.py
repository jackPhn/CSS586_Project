# %%
import pathlib
import sys
import numpy as np
import pypianoroll
import matplotlib.pyplot as plt
from tensorflow.keras import utils, layers, Model, optimizers, Sequential

sys.path.append("..")
from musiclearn import config