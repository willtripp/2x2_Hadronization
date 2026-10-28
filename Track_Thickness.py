"""
Created by W. M. Tripp on October 12st, 2026

The goal of this script is to calculate a meaningful measure of thickness for reconstructed tracks based on a vector
going from start to end and each deposit's distance to that line
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SPINE_Reader import load_driver
from plot_data_mc import plot_data_mc
## ----- Function to calculate the distance of each deposit ----- ##
def calc_dist(track_start, track_end, points):
    '''
    Parameters
    ----------
    track_start: Start point of the track
    track_end: End point of the track
    points: an array of all the voxels lit up in the track

    Returns:
    -------
    distances: an array of all the scalar distances from a voxel to the closest point along the track
    '''

    ## ----- Create a bunch of arrays ---- ##
    start     = np.array(track_start)
    end       = np.array(track_end)
    track     = end - start
    track_dir = track / np.linalg.norm(track)   #unit vector

    #Find vector from start to deposit to make a projection.
    u = points - start

    track_length_sq = np.dot(track, track)

    # if the track is just a point the math is easy
    if track_length_sq == 0:
        return np.array([np.linalg.norm(u)])

    # find the ratio along the track of the closest point
    # think I may actually need an outer product here but I should double check
    proj = (np.dot(u, track) / track_length_sq)

    # then we just multiply that ratio times our track vector for the desired point
    # closest_point = start + proj * track
    closest_points = start + np.multiply.outer(proj, track)

    # print(closest_points.shape)

    distances = np.linalg.norm(points - closest_points, axis=-1)
    # print(distances.shape)
    return distances

# def bragg_dep():

def get_stats(distances):
    '''

    Parameters
    ----------
    distances: Feed in an array of the distance from a hit to the track vector for every point of the particle

    Returns
    -------
    stats: Returns average, max, and standard deviation
    '''

    mean = distances.mean()
    max  = distances.max()
    std  = distances.std()

    return mean, max, std

def process_particle(entry , particle, is_truth=False):
    '''
    A single function to get the relevant info for each particle rather than having to change it for data, mc_reco, and mc_truth...

    Parameters
    ----------
    particle: a particle object either
    truth: is this a truth particle or a reco object

    Returns: a data frame to be concactenated on to the main frame
    -------
    '''
    #get points
    points = np.array(particle.points)

    #get distance
    start     = particle.start_point
    end       = particle.end_point
    distances = calc_dist(start, end, points)

    mean = distances.mean()
    max  = distances.max()
    std  = distances.std()

    containment = particle.is_contained

    length = particle.tr

    if is_truth:
        df = pd.DataFrame([{"entry": entry, "truth_mean": mean, "truth_max": max, "truth_std": std, "truth_containment": containment}])
    else:
        df = pd.DataFrame([{"entry": np.array(entry), "mean": np.array(mean), "max": np.array(max), "std": np.array(std), "is_contained": containment}])

    return df


# load in the driver
driver = load_driver("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/*.hdf5")

# One Mega list of all the distances
entry       = []
interaction = []

data_dists  = []
data_means  = []
data_maxs   = []
data_stds   = []
data_pids   = []

containment = []    #will for sure be useful...

data_df = pd.DataFrame()

for entry in range(len(driver)):
    data = driver.process(entry)
    trig_type = data.get("trigger", [])
    trig_type = trig_type.type
    if trig_type != 5:
        continue
    particles = data.get("reco_particles", [])
    for p in particles:
        if p.shape != 1:
            continue
        data_df = pd.concat([data_df, process_particle(entry, p)], ignore_index=True)

## ----- Monte Carlo Time ----- ##
MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/*.hdf5")

mc_df = pd.DataFrame()

for entry in range(len(MC_driver)):
    mc       = MC_driver.process(entry)
    pairs    = mc.get("particle_matches_r2t", [])
    overlaps = np.asarray(mc.get("particle_matches_r2t_overlaps", np.ones(len(pairs))))

    for i, (reco_p, truth_p) in enumerate(pairs):
        if reco_p.shape != 1:
            continue
        mc_df = pd.concat([mc_df, process_particle(entry, reco_p)], ignore_index=True)

plot_data_mc(
    data_df, mc_df,
    column = "mean",
    bins   = np.linspace(0, 4, 20),
    xlabel = "Distance from dep voxel to track [mm]",
    title  = "Track thickness data / mc",
    normalize_mc_to_data = True,
    outfile=("plots/avg_dep_dist_full.png"),
)