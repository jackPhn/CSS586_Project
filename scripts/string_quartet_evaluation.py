# %%
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append("..")
from musiclearn import config, processing, vae_models

# %%
exp_name = "mtvae"
exp_time = "2021-06-06T09:32:32"


# %% [markdown]
# ## Metrics
#
# (from Yang and Lerch 2020)
#
# - Pitch count
# - Pitch class histogram
# - Pitch class transition matrix
# - Pitch range
# - Average pitch interval
# - Note count
# - Average inter-onset interval
# - Note length histogram
# - Note length transition matrix
