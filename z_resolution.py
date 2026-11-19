"""
Created by W. M. Tripp on November 5th 2026

How well could we possibly reconstruct z?

1. Select out interactions with containd HADRONS...
    Should I include pi-zeros?

2. Sum up all the visible hadronic energy....

3. Take highest energy hadron...

4. Would also be good to know how many events have only one uncontained hadron... This would bias our data

plots:
1) truth and reco z distribution
2) z resolution
3) hadronic energy resolution:
    Both using info available to the reco (so not neutrons, etc...) and using the neutrino + lepton energy....
4) leading hadron energy resolution
"""

import numpy as np
import pandas as pd
from numba.core.cgutils import false_bit

from plot_data_mc import plot_data_mc, plot_mc_only
from plot_data_mc import plot_2d_hist

from SPINE_Reader import load_driver

#need to make sure the topology is readable
def clean_topology(t):
    return str(t).strip().strip("'\"")

def est_transferred_energy(interaction):
    #TODO: figure out if we should include electrons here to get a better handle on transferred energy rather than ehadvis
    e_trans = 0.0
    particles = interaction.particles
    for p in particles:
        #skip leptons
        if p.pid == 1 or p.pid ==2:
            continue
        # assume photons come form hadrons (or at least transferred energy)
        if p.pid == 0:
            e_trans += p.ke
        # assume protons already existed
        if p.pid == 4:
            e_trans += p.ke
        # assume everything else was created by interaction
        else:
            e_trans += np.sqrt(p.mass ** 2 + p.p ** 2)
    # print(e_trans)

    return e_trans

def est_hadronic_energy(reco_interaction, truth_interaction):
    '''
    We're going to fude the reco here a bit to get an upper bound on how well we can get the hadronic energy
    i.e. we're going to use truth pids but then use the reco energies... but the truth masses (at least for hadrons)
    Well... I guess we can use the best match but that may introduce problems for fragmented particles
    TODO: A possible extension could be to enforce a cut on overlaps
    Returns
    -------

    '''




def get_reco_z(reco_interaction):
    particles = reco_interaction.particles

    approx_energy_transfer = est_transferred_energy(reco_interaction)

    #find leading charged hadron
    max_had_E           = 0.0

    for p in particles:
        if p.shape != 1:
            continue
        if p.pid == 2:
            continue

        #grab whatever spine thinks is the best ke (either range or calo)
        ke       = p.ke
        momentum = p.p
        mass     = p.mass

        if p.is_primary != True:
            continue

        E = np.sqrt(mass ** 2 + momentum ** 2)

        if E > max_had_E:
            max_had_E = E

    if approx_energy_transfer == 0.0:
        return np.nan

    z = max_had_E / approx_energy_transfer
    return z

def get_truth_z(truth_interaction):
    W = truth_interaction.hadronic_invariant_mass
    x = truth_interaction.bjorken_x
    y = truth_interaction.inelasticity
    energy_transfer = truth_interaction.energy_transfer

    n_hadrons = 0

    max_E = 0.0
    for p in truth_interaction.particles:
        if p.shape != 1:
            continue
        if p.pid == 2:
            continue
        if p.is_primary != True:
            continue

        E = p.energy_init

        if E > max_E:
            max_E = E
            id = p.id


        n_hadrons +=1

    z = max_E / energy_transfer

    # if n_hadrons == 0:
    #     return np.nan

    return z

def get_containment(interaction, had_only = True):
    '''
    Function to decide if tha hadrons are contained or not. We don't care about anything else
    update: we do care a lot about pi0 decays I think
    '''
    particles = interaction.particles
    # fixed!: I think this means only the first particle needs to be contained. Make this all
    if had_only == True:
        for p in particles:
            if p.shape != 1:
                continue
            elif p.pid == 2:
                continue
            elif p.is_contained == False:
                return False
        return True
    if had_only == False:
        for p in particles:
            if p.pid == 2:
                continue
            #don't care about deltas or micheles
            if p.shape == 3 or p.shape == 2:
                continue
            if p.is_contained == False:
                return False
        return True

MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/MiniRun6.5_1E19_RHC.spine.*.MLRECO_SPINE.hdf5")

entry_list   = []
reco_z       = []
truth_z      = []
reco_id      = []
truth_id     = []
reco_topology = []
truth_topology = []


for entry in range(len(MC_driver)):
    mc = MC_driver.process(entry)
    pairs = mc["interaction_matches_r2t"]
    overlaps = mc["interaction_matches_r2t_overlap"]

    # entry_list.append(entry)

    #fixme: change these cuts to be based on truth info where we could reasonably get better at reconstructing something

    for i, (reco_int, truth_int) in enumerate(pairs):
        if overlaps[i] < 0.5:
            continue
        #fixme: could be one line
        if get_containment(reco_int) == False:
            continue
        # if get_containment(truth_int, had_only=False) == False:
            continue
        if clean_topology(reco_int.topology) == '': #or clean_topology(truth_int.topology) == '':
            continue
        if clean_topology(reco_int.topology) == '1mu': # or clean_topology(truth_int.topology) == '1mu':
            continue
        if reco_int.is_fiducial != True: #or truth_int.is_fiducial != True:
            continue
        if len(reco_int.particles) == 1:
            if reco_int.particles[0].pid == 2:
                continue

        assert reco_int.is_fiducial #and truth_int.is_fiducial

        entry_list.append(entry)
        reco_id.append(reco_int.id)
        truth_id.append(truth_int.id)
        reco_topology.append(reco_int.topology)
        truth_topology.append(truth_int.topology)

        reco_z.append(get_reco_z(reco_int))
        truth_z.append(get_truth_z(truth_int))

        # truth_z.append(get_truth_z(truth_int))


resolution = np.asarray(reco_z) - np.asarray(truth_z)


# print(entry_list)


df = pd.DataFrame({"entry": entry_list,
                   "reco_id": reco_id,
                   "truth_is": truth_id,
                   "reco_topology": reco_topology,
                   "truth_topology": truth_topology,
                   "reco_z": reco_z,
                   "truth_z": truth_z,
                   "resolution": resolution})


df.to_csv("csv_files/z_resolution_reco_cuts_only.csv", index=False)

plot_mc_only(
    df,
    column="resolution",
    bins=np.linspace(-5, 5, 70),
    xlabel="z Resolution (reco - truth)",
    title="z resolution",
    resolution=True,
    outfile="plots/z_resolution_reco_cuts_only.png",
)

plot_mc_only(
    df,
    column=["truth_z", "reco_z"],
    bins=np.linspace(0, 5, 50),
    xlabel="z",
    title="Leading Charged Hadron Energy / Transferred Energy",
    outfile="plots/z_distribution_reco_cuts_only.png",
)
