import numpy as np
import matplotlib.pyplot as plt

from SPINE_Reader import load_driver
from plot_data_mc import plot_data_mc

from dEdx import get_proj_dists
from dEdx import segmented_dEdx

from had_functions import get_best_pid

driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/*")


# for entry in driver:
#     data = driver.process(entry=entry)
#     interactions = data.get("reco_interactions", [])


# cherry pick a nice event
data = driver.process(entry=0)
particles = data.get("reco_particles", [])

print(get_best_pid(particles[8], data["truth_particles"]))