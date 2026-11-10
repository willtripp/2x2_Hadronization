"""
Created by W.M. Tripp on November 9th, 2026
"""

##TODO: Fix the handling of leftover segments smaller than 0.25cm

import numpy as np
import pandas as pd
from plot_data_mc import plot_data_mc
from plot_data_mc import plot_2d_hist

from SPINE_Reader import load_driver

def get_proj_dists(start_point, end_point, points):
    '''
    A function that returns the distance along a track a given point lies
    Parameters
    ----------
    track: vector of the particle track
    points: The voxel coordinate of an energy deposition

    Returns
    -------
    '''

    #build track
    track  = end_point - start_point
    length = np.linalg.norm(track)

    #vector from start to deposition
    u = points - start_point

    #calculate the length along the track at which the deposition lies
    proj_dists = np.dot(u, track) / length

    return proj_dists

def segmented_dEdx(particle, segment_length):
    '''
    A function that chops up a track into desired dx lengths and calculates dEdx for that segment
    Parameters
    ----------
    particle: a reco particle
    segment_length: the desired dx length

    Returns
    -------

    '''
    length      = particle.length
    depositions = particle.depositions

    #first n steps of
    n_steps = int(np.floor(length / segment_length))

    dEdx      = np.ones(n_steps+1)
    dep_res_range = np.ones(n_steps+1)

    len = 0.0

    proj_dists = np.array(get_proj_dists(particle.start_point, particle.end_point, particle.points))

    for i in range(n_steps):

        mask = (len <= proj_dists) & (proj_dists < (len + segment_length))
        len += segment_length
        # mask[proj_dists < len]# + segment_length))

        # len += segment_length

        masked_depositions = depositions[mask]

        dEdx[i]  = masked_depositions.sum() / segment_length
        dep_res_range[i] = length - len

    #treat any remaining dist that didn't divide nicely by two.
    #toss out values less than 0.25 to avoid skewing the data with low denominators
    if length - len < 0.25:
        dEdx[-1]  = -1
        dep_res_range[-1] = -1
    else:
        mask               = proj_dists >= len
        masked_depositions = depositions[mask]

        dEdx[-1]  = masked_depositions.sum() / (length - len)
        dep_res_range[-1] = 0

    return dEdx, dep_res_range


if __name__ == "__main__":
    driver = load_driver("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/*.hdf5")
    # driver = load_driver("/Users/williamtripp/Desktop/DUNE/data/Sandbox/SPINE/packet-0050017-2024_07_09_00_04_33_CDT.MLRECO_SPINE.hdf5")

    dEdx_list      = []
    dep_range_list = []
    length_list    = []

    for entry in range(len(driver)):
        data = driver.process(entry)
        trig_type = data["trigger"]
        trig_type = trig_type.type
        if trig_type != 5:
            continue

        particles = data["reco_particles"]
        for p in particles:
            if p.shape != 1:
                continue
            if p.is_contained != True:
                continue

            dEdx, dep_range = segmented_dEdx(p, 2.0)
            # length = p.length

            dEdx_list.extend(dEdx)
            dep_range_list.extend(dep_range)
            # length_list.append(length)

    # length_list = np.asarray(length_list)
    # residual_range_list = np.asarray(length_list) - np.asarray(dep_range_list)

    data_df = pd.DataFrame({"dEdx": dEdx_list, "dep_range": dep_range_list})#, "length": length_list})

    ## ----- MC Time ----- ##
    MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/MiniRun6.5_1E19_RHC.spine.000000*.MLRECO_SPINE.hdf5")
    # MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/MiniRun6.5_1E19_RHC.spine.0000000.MLRECO_SPINE.hdf5")

    dEdx_list      = []
    dep_range_list = []
    length_list    = []

    for entry in range(len(MC_driver)):
        mc = MC_driver.process(entry)
        pairs = mc.get("particle_matches_r2t", [])
        overlaps = np.asarray(mc.get("particle_matches_r2t_overlaps", np.ones(len(pairs))))

        for i, (reco_p, truth_p) in enumerate(pairs):
            if reco_p.shape != 1:
                continue
            if reco_p.is_contained != True:
                continue

            dEdx, dep_range = segmented_dEdx(reco_p, 2.0)
            # length = reco_p.length

            dEdx_list.extend(dEdx)
            dep_range_list.extend(dep_range)
            # length_list.append(length)

    mc_df = pd.DataFrame({"dEdx": dEdx_list, "dep_range": dep_range_list})#, "length": length_list})

    print(data_df)

    print(data_df.shape, mc_df.shape)
    print(data_df.dtypes)
    print(data_df.head())

    plot_data_mc(
        data_df, mc_df,
        column = "dEdx",
        bins   = np.linspace(0, 15, 50),
        xlabel = "dE/dx [MeV/cm]",
        title  = "Track dE/dx from 2cm segments",
        normalize_mc_to_data = True,
        outfile=("plots/segmented_dEdx_from_2cm.png")
    )

    plot_2d_hist(
        data_df,
        x_column="dep_range",
        y_column="dEdx",
        x_bins=np.linspace(0, 120, 50),
        y_bins=np.linspace(0, 20, 50),
        xlabel="Residual Range [cm]",
        ylabel="dE/dx [MeV/cm]",
        title="Data dE/dx for Contained Tracks",
        outfile=("plots/Data_Track_dEdcResRange.png")
    )

    plot_2d_hist(
        mc_df,
        x_column="dep_range",
        y_column="dEdx",
        x_bins=np.linspace(0, 120, 50),
        y_bins=np.linspace(0, 20, 50),
        xlabel="Residual Range [cm]",
        ylabel="dE/dx [MeV/cm]",
        title="MC dE/dx for Contained Tracks",
        outfile=("plots/MC_Track_dEdxResRange.png")
    )