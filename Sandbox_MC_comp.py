"""
Created by W.M Tripp on October 19th, 2026

The goal here is to coalse previois scripts into a newplace that can examine MC and the sandbox set at the same time

A few things I want from this scipt are

1. Selection based on the trigger info in sandbox data
    5 is beam and 6 is cosmic. It probably makes the most sense to keep both for the time being

2. simultaneous running of mc and data loading...

3. somwhat easy extension into larger datasets

4. save csv outputs and produce plots
"""

from pathlib import Path

import pandas as pd
import yaml
import time
import csv
import argparse
import sys

from spine.driver import Driver
from spine_tools/SPINE_Reader import get_nonzero_entries    #fixme: modify to be triggered entries
from spine_tools/SPINE_Reader import reduced_data_loader
from multiprocessing import Process

## ----- Step 0: Set parameters... Local running only for now ----- ##
#fixme: this is a function now... see the top of SPINE_reader.py

data_path = Path("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/*.hdf5")
config    = Path("spine_tools/SPINE_Reader.yaml")
cfg_txt   = config.read_text().replace("DATA_PATH", str(data_path))
cfg       = yaml.safe_load(cfg_txt)
driver    = Driver(cfg)

attributes = [
    "entry",
    "pid",
    "id",
    "interaction_id",
    "shape",
    "size",
    "length",
    "is_contained",
    "start_point",
    "end_point",
     "depositions_sum",
    "pid_scores",
    "start_dir",
    "end_dir",
    "volume_id",
    "is_valid",
]

