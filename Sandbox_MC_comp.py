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
from SPINE_Reader import iter_trigger
from SPINE_Reader import load_MC_reco_truth
from multiprocessing import Process

## ----- Step 0: Set parameters... Local running only for now ----- ##
#fixme: this is a function now... see the top of SPINE_reader.py

data_path = Path("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/*.hdf5")
config    = Path("spine_tools/read_spine_hdf5.yaml")
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

iter_trigger(driver, "csv_files/sandbox_beam_trigger.csv", attributes, attributes)

MC_PATH    = Path("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/*.hdf5")
# MC_PATH    = Path("/cluster/tufts/lblnu/dune/data/users/wtripp/2x2/MiniRun6.5/SPINE/*.hdf5")
CONFIG     = Path("spine_tools/read_spine_hdf5.yaml")
mc_cfg_txt    = CONFIG.read_text().replace("DATA_PATH", str(MC_PATH))
mc_cfg        = yaml.safe_load(cfg_txt)

mc_driver = Driver(mc_cfg)

base_attributes = [
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
        "start_dir",
        "end_dir",
        "volume_id",
        "is_valid",
    ]
all_attributes = [
        "entry",
        "true_pid",
        "reco_pid",
        "true_id",
        "reco_id",
        "true_interaction_id",
        "reco_interaction_id",
        "true_shape",
        "reco_shape",
        "true_size",
        "reco_size",
        "true_length",
        "reco_length",
        "true_is_contained",
        "reco_is_contained",
        "true_start_point",
        "reco_start_point",
        "true_end_point",
        "reco_end_point",
        "true_depositions_sum",
        "reco_depositions_sum",
        "reco_pid_scores",
        "overlap",
        "true_start_dir",
        "reco_start_dir",
        "true_end_dir",
        "reco_end_dir",
        "true_volume_id",
        "reco_volume_id",
        "true_is_valid",
        "reco_is_valid",
    ]

outfile = "csv_files/MC_reco_truth_match.csv"

load_MC_reco_truth(mc_driver, outfile, base_attributes, all_attributes)

# ## ----- Dataframes and plotting ---- ##
# '''
# shoudl this be a seperate script? Probably...
# '''
# data_df = pd.read_csv("csv_files/sandbox_beam_trigger.csv")
# mc_df = pd.read_csv("csv_files/MC_reco_truth_match.csv")

