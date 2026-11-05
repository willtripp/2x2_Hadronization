import numpy as np

from SPINE_Reader import load_driver
from plot_data_mc import plot_data_mc

from Track_Thickness import bragg_dQdx

driver = load_driver("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/packet-0050017-2024_07_09_00_04_33_CDT.MLRECO_SPINE.hdf5")

# cherry pick a nice event
data = driver.process(entry=74)
particles = data.get("reco_particles", [])

# chosoe a nice contained proton
p = particles[31]

print(bragg_dQdx(p))