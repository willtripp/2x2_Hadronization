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
from plot_data_mc import plot_2d_hist
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

def get_dQdx(particle):
    dx          = particle.length
    depositions = np.array(particle.depositions)
    dQ          = depositions.sum()

    return dQ / dx

def get_bragg_dQdx(particle):
    '''
    A function to find the dQ/dx depositions for the bragg peak. In this case it considers the last 10c of the track.
    Note: particles must be contained
    Parameters
    ----------
    points
    depositions

    Returns
    -------

    '''
    if particle.is_contained == False:
        #TODO: The end point must be contained, we actauilly don't care if it starts outside...
        raise ValueError("Particle must be contained")
    if particle.shape != 1:
        raise ValueError("Particle must be a Track")

    start_point = np.array(particle.start_point)
    end_point   = np.array(particle.end_point)
    track       = end_point - start_point
    points      = np.array(particle.points)
    depositions = np.array(particle.depositions)
    length      = particle.length

    # If a particle travels less than 10cm the whole thing is a bragg peak
    if length < 10:
        dQdx = depositions.sum() / length
        return dQdx

    # If not then we need to scale
    frac = 10 / length

    # set a threshold for the projection
    threshold  = 1 - frac

    #calculate the projection onto the track as a fraction
    u = np.array(particle.points) - start_point

    proj = (np.dot(u, track) / (length * length))

    # print(len(proj))
    # print(proj)

    proj[proj < threshold] = 0
    proj[proj >= threshold] = 1

    # print(proj)

    mask = proj > 0

    #fixme: ahahaha these dimensions don't make any sense lol
    last_10cm_points = points[mask]
    last_10cm_depositions = depositions[mask]

    if last_10cm_depositions.size == 0:
        return -1

    # print(last_10cm_points)
    # print(last_10cm_depositions)

    # dx = np.sum(np.linalg.norm(np.diff(last_10cm_points, axis=0), axis=1))
    dx = np.linalg.norm(end_point - last_10cm_points[0])
    # print(dx)

    dQ = np.sum(last_10cm_depositions)

    return dQ / dx

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

    length = particle.length

    dQdx = get_dQdx(particle)

    bragg_dQdx = get_bragg_dQdx(particle)

    if is_truth:
        df = pd.DataFrame([{"entry": entry,
                            "truth_mean": mean,
                            "truth_max": max,
                            "truth_std": std,
                            "truth_containment": containment,
                            "truth_dQdx": dQdx,
                            "truth_bragg_dQdx": bragg_dQdx,}])
    else:
        df = pd.DataFrame([{"entry": np.array(entry),
                            "mean": np.array(mean),
                            "max": np.array(max),
                            "std": np.array(std),
                            "is_contained": containment,
                            "dQdx": dQdx,
                            "bragg_dQdx": bragg_dQdx,}])

    return df

if __name__ == "__main__":
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
            if p.is_contained != True:
                continue
            data_df = pd.concat([data_df, process_particle(entry, p)], ignore_index=True)

    ## ----- Monte Carlo Time ----- ##
    MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/MiniRun6.5_1E19_RHC.spine.000000*.MLRECO_SPINE.hdf5")

    mc_df = pd.DataFrame()

    for entry in range(len(MC_driver)):
        mc       = MC_driver.process(entry)
        pairs    = mc.get("particle_matches_r2t", [])
        overlaps = np.asarray(mc.get("particle_matches_r2t_overlaps", np.ones(len(pairs))))

        for i, (reco_p, truth_p) in enumerate(pairs):
            if reco_p.shape != 1:
                continue
            if reco_p.is_contained != True:
                continue
            mc_df = pd.concat([mc_df, process_particle(entry, reco_p)], ignore_index=True)

    # plot_data_mc(
    #     data_df, mc_df,
    #     column = "mean",
    #     bins   = np.linspace(0, 4, 20),
    #     xlabel = "Distance from dep voxel to track [cm]",
    #     title  = "Track thickness data / mc",
    #     normalize_mc_to_data = True,
    #     outfile=("plots/avg_dep_dist.png"),
    # )

    plot_data_mc(
        data_df, mc_df,
        column = "dQdx",
        bins = np.linspace(0, 10, 10),
        xlabel = "dQ/dx [ke / cm]",
        title = "Track dQ/dx",
        normalize_mc_to_data= True,
        outfile=("plots/Track_dqdx_test.png"),
    )

    plot_data_mc(
        data_df, mc_df,
        column="bragg_dQdx",
        bins=np.linspace(0, 10, 10),
        xlabel="dQ/dx [ke / cm]",
        title="Track dQ/dx for last 10cm (Bragg Peak)",
        normalize_mc_to_data=True,
        outfile=("plots/Track_bragg_dqdx_test.png"),
    )
