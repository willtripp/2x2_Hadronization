import numpy as np
import matplotlib.pyplot as plt

from SPINE_Reader import load_driver
from plot_data_mc import plot_data_mc

from dEdx import get_proj_dists
from dEdx import segmented_dEdx

driver = load_driver("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/packet-0050017-2024_07_09_00_04_33_CDT.MLRECO_SPINE.hdf5")


# for entry in driver:
#     data = driver.process(entry=entry)
#     interactions = data.get("reco_interactions", [])


# cherry pick a nice event
data = driver.process(entry=74)
particles = data.get("reco_particles", [])

# choose a nice contained proton
p = particles[31]

start_point = p.start_point
end_point = p.end_point

points = np.array(p.points)

first_point = points[0]
last_point = points[-1]

dEdx, dep_range = segmented_dEdx(p, 2)

print(dEdx)
print(dep_range)

plt.plot(dep_range, dEdx)
plt.savefig("plots/segmented_dEdx-scratch.png")